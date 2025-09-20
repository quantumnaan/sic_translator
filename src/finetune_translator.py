# finetune_translator.py

"""
finetune_translator.py (フェーズC)

目的:
フェーズA, Bで構築したモデルをエンコーダとして統合し、事前学習済みの
言語モデルをデコーダとして接続する。
(音声, 英語テキスト)の対訳データセットを用いて、このエンドツーエンドの
翻訳モデル全体をファインチューニングする。
"""

import warnings
from transformers.utils.logging import set_verbosity_error

# 1. transformersライブラリ自体のワーニングを抑制
# これにより、多くのtransformers関連のメッセージが非表示になります
set_verbosity_error()

# 2. その他のライブラリ（torchaudioなど）のワーニングを抑制
# FutureWarningとUserWarningを無視するように設定
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import glob
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
import joblib

from transformers import (
    Wav2Vec2Model,
    AutoTokenizer,
    AutoModelForCausalLM,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Wav2Vec2FeatureExtractor,
    PreTrainedModel,
    Wav2Vec2Config,
    DataCollatorWithPadding,
    AutoConfig,
)
from transformers.modeling_outputs import BaseModelOutput

from model import AcousticUnitEncoder, MultimodalAcousticModel  # フェーズBのmodel.pyからインポート
# finetune_translator.py (e.g., after the imports)

from dataclasses import dataclass
from typing import Any, Dict, List, Union


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune the full translation model.")
    parser.add_argument("--acoustic_model_path", type=str, required=True)
    parser.add_argument("--semantic_model_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--parallel_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--decoder_model_name", type=str,
                        default="facebook/bart-base")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume training from the latest checkpoint in output_dir."
    )
    return parser.parse_args()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads the inputs received.
    Args:
        feature_extractor ([`Wav2Vec2FeatureExtractor`])
            The feature extractor used for proccessing the data.
        tokenizer ([`PreTrainedTokenizer`])
            The tokenizer used for encoding the labels.
    """
    feature_extractor: Any
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        # and need different padding methods.
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        # Pad the audio inputs
        batch = self.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        # Pad the text labels
        labels_batch = self.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


# --- Encoder: 音声から意味表現へ ---

class SpeechToMeaningEncoder(PreTrainedModel):
    # PreTrainedModelとの互換性のために、config_classを指定
    config_class = Wav2Vec2Config

    # Trainerが入力キーを認識できるように設定
    main_input_name = "input_values"

    def __init__(self, config, decoder_config, acoustic_model_path, semantic_model_path, params_path):
        super().__init__(config)

        # 1. フェーズAモデルのロード
        self.acoustic_base_model = Wav2Vec2Model.from_pretrained(
            acoustic_model_path)
        kmeans_path = os.path.join(
            acoustic_model_path, "kmeans_codebook.joblib")
        self.kmeans_model = joblib.load(kmeans_path)

        # 2. フェーズBモデルのロード
        with open(params_path, 'r') as f:
            best_params = json.load(f)


        # 💥【修正点】MultimodalAcousticModelをロードしてから、audio_encoder部分のみを使用
        # この部分は直接ロードせず、下で重みを読み込む
        self.semantic_encoder = AcousticUnitEncoder(
            vocab_size=best_params.get('vocab_size', 512),
            embedding_dim=best_params['embedding_dim'],
            hidden_dim=best_params['embedding_dim'],
            num_layers=best_params['audio_encoder_layers']
        )
        # 親モデルのstate_dictをロードし、プレフィックスが一致する部分だけを抽出してロードする
        full_state_dict = torch.load(semantic_model_path, map_location="cpu")
        audio_encoder_state_dict = {
            key.replace("audio_encoder.", ""): value
            for key, value in full_state_dict.items()
            if key.startswith("audio_encoder.")
        }
        self.semantic_encoder.load_state_dict(audio_encoder_state_dict)
        self.projection = nn.Linear(best_params['embedding_dim'], decoder_config.d_model)

        # 3. 学習中はエンコーダの重みを固定（凍結）
        # 新しい提案（音響モデルのみを凍結し、意味モデルとアダプターは学習対象にする）
        print("Freezing acoustic_base_model parameters...")
        for param in self.acoustic_base_model.parameters():
            param.requires_grad = False
        
        # semantic_encoder と projection はデフォルトで requires_grad=True なので学習される
        # 明示的に書くなら以下の通り
        print("Setting semantic_encoder and projection parameters to be trainable...")
        for param in self.semantic_encoder.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True

    def forward(self, input_values=None, **kwargs):
        # 💥【最重要修正点】self.deviceに頼らず、入力テンソルからデバイスを取得する
        device = input_values.device

        # 各サブモジュールが同じデバイスにあることを確認
        self.acoustic_base_model.to(device)
        self.semantic_encoder.to(device)

        with torch.no_grad(): # acoustic_base_modelは凍結されているので、この部分は勾配計算不要
            features = self.acoustic_base_model(input_values).last_hidden_state

        semantic_outputs = []
        for i in range(features.shape[0]):  # バッチ内の各サンプルを処理
            feats_np = features[i].cpu().detach().numpy()

            # 特徴量 -> 音響単位系列
            units = self.kmeans_model.predict(feats_np)
            units_tensor = torch.LongTensor(units).unsqueeze(0).to(device)

            # 音響単位系列 -> 意味ベクトル（LSTMの全系列の隠れ状態を取得）
            semantic_output = self.semantic_encoder.lstm(
                self.semantic_encoder.embedding(units_tensor))[0]
            semantic_outputs.append(semantic_output)

        # 注意: バッチ内のシーケンス長が異なるため、パディングが必要
        # ここではtransformersのEncoderDecoderModelが扱いやすいように最終的な隠れ状態のみを返す
        # より高度な実装ではここでパディング処理を行う
        final_hidden_states = torch.cat(
            [s.mean(dim=1) for s in semantic_outputs]).unsqueeze(1)

        with torch.enable_grad(): # アダプター層の勾配計算を有効にする
            projected_states = self.projection(final_hidden_states)
        # EncoderDecoderModelが必要とする形式で、'last_hidden_state'キーと共に返す
        return BaseModelOutput(last_hidden_state=projected_states)


# --- データ準備 ---
class ParallelDataset(Dataset):
    def __init__(self, data_dir, feature_extractor, tokenizer, max_text_length=128):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

        audio_dir = os.path.join(data_dir, "audio")
        text_dir = os.path.join(data_dir, "text")
        self.pairs = []
        for audio_path in glob.glob(os.path.join(audio_dir, "*.wav")):
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            text_path = os.path.join(text_dir, f"{base_name}.txt")
            if os.path.exists(text_path):
                self.pairs.append({"audio": audio_path, "text": text_path})

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # 1. 音声のロードと前処理
        waveform, sr = torchaudio.load(pair["audio"])
        # feature_extractorはリストではなく単一の波形を受け取れる
        processed_audio = self.feature_extractor(
            waveform.squeeze(0).numpy(),  # NumPy配列として渡す
            sampling_rate=self.feature_extractor.sampling_rate,
        )

        # 2. テキストのロードとトークン化
        with open(pair["text"], 'r', encoding='utf-8') as f:
            text = f.read().strip()

        tokenized_text = self.tokenizer(
            text,
            padding="max_length",  # ここでパディング
            max_length=self.max_text_length,
            truncation=True,
        )

        # Trainerが必要とするキー名で辞書を返す
        return {
            "input_values": torch.tensor(processed_audio.input_values[0]),
            "labels": torch.tensor(tokenized_text.input_ids)
        }

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """
    予期しない引数がモデルに渡されるのを防ぐためのカスタムTrainer
    """
    # 💥【修正点】Trainer本体からの呼び出しに合わせ、引数に num_items_in_batch を追加
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # このメソッドは num_items_in_batch を受け取りますが、
        # 親クラスのメソッドには渡さないことで、ここで引数を安全に「握りつぶし」ます。
        # 親クラスの compute_loss を呼び出す際は、余分な引数を渡さない
        return super().compute_loss(model, inputs, return_outputs)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 各コンポーネントをロード
    encoder_config = Wav2Vec2Config.from_pretrained(args.acoustic_model_path)
    decoder_config = AutoConfig.from_pretrained(args.decoder_model_name)
    encoder = SpeechToMeaningEncoder(
        config=encoder_config,
        decoder_config=decoder_config,
        acoustic_model_path=args.acoustic_model_path,
        semantic_model_path=args.semantic_model_path,
        params_path=args.params_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_name)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.acoustic_model_path)

    # 2. Encoder-Decoderモデルを構築
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=None,  # カスタムエンコーダを使用
        decoder_pretrained_model_name_or_path=args.decoder_model_name,
        encoder_model=encoder
    )
    # 特殊トークンの設定
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    model.tie_weights()

    # 3. データセットとデータコレータを準備
    dataset = ParallelDataset(args.parallel_data_dir,
                              feature_extractor, tokenizer)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    # 4. 学習引数とTrainerを設定
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        save_strategy="epoch",
        logging_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 5. ファインチューニング実行
    print("\n--- Starting Final Fine-tuning (Phase C) ---")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("\n--- Fine-tuning Complete ---")
    

    # 6. 最終モデルの保存
    trainer.save_model(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    print(f"Final translator model saved to '{args.output_dir}'")


if __name__ == "__main__":
    args = get_args()
    main(args)
