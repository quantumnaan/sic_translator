# src/demo_translator.py

"""
demo_translator.py

目的:
フェーズCでファインチューニングされた、エンドツーエンドの音声翻訳モデルを使い、
ターゲット言語の単一の音声ファイルを入力として受け取り、
その英語翻訳結果をテキストで出力するデモ。
"""

import argparse
import torch
import torchaudio
from transformers import (
    EncoderDecoderModel,
    Wav2Vec2FeatureExtractor,
    AutoTokenizer,
)

def main():
    parser = argparse.ArgumentParser(
        description="Translate a single audio file using the final fine-tuned model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the final translator model directory (output of Phase C).",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to the target language audio file to be translated.",
    )
    args = parser.parse_args()

    # デバイスの決定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 完成したモデルと関連コンポーネントをロード
    print(f"Loading model from '{args.model_path}'...")
    try:
        # EncoderDecoderModelは、カスタムエンコーダとデコーダの両方を含む
        # 学習済みの完全なモデルです。
        model = EncoderDecoderModel.from_pretrained(args.model_path).to(device)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model path is correct and contains all necessary files "
              "(pytorch_model.bin, config.json, preprocessor_config.json, etc.).")
        return

    model.eval() # 推論モードに設定

    # 2. 翻訳したい音声ファイルをロードして前処理
    print(f"Loading and processing audio file: '{args.audio_file}'...")
    try:
        waveform, sample_rate = torchaudio.load(args.audio_file)

        # サンプリングレートがモデルの期待するものと違う場合はリサンプリング
        if sample_rate != feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=feature_extractor.sampling_rate
            )
            waveform = resampler(waveform)

        # feature_extractor を使ってモデルへの入力形式に変換
        inputs = feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(device)

    except Exception as e:
        print(f"Error processing audio file: {e}")
        return

    # 3. モデルによる翻訳の生成（推論）
    print("Translating...")
    with torch.no_grad():
        # .generate() メソッドで翻訳結果（トークンID）を生成
        predicted_ids = model.generate(input_values)

    # 4. 生成されたトークンIDをテキストにデコード
    # skip_special_tokens=True で、<s> や </s> といった特殊トークンを除外
    transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # 5. 結果の表示
    print("\n" + "="*50)
    print("Translation Result")
    print("="*50)
    print(f"Input Audio: {args.audio_file}")
    print(f"Translated Text: {transcription}")
    print("="*50)


if __name__ == "__main__":
    main()