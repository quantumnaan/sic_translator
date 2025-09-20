# src/debug_encoder.py (Corrected Loading Logic)

import argparse
import torch
import torchaudio
import joblib
import json
import os
import numpy as np
from safetensors.torch import load_file
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, PreTrainedModel, Wav2Vec2Config, AutoConfig
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from model import AcousticUnitEncoder

# SpeechToMeaningEncoderクラスの定義は変更なし（finetune_translator.pyからコピー）
class SpeechToMeaningEncoder(PreTrainedModel):
    config_class = Wav2Vec2Config
    main_input_name = "input_values"
    def __init__(self, config, decoder_config, acoustic_model_path, semantic_model_path, params_path):
        super().__init__(config)
        self.acoustic_base_model = Wav2Vec2Model.from_pretrained(acoustic_model_path)
        kmeans_path = os.path.join(acoustic_model_path, "kmeans_codebook.joblib")
        self.kmeans_model = joblib.load(kmeans_path)
        with open(params_path, 'r') as f: best_params = json.load(f)
        self.semantic_encoder = AcousticUnitEncoder(vocab_size=best_params.get('vocab_size', 512), embedding_dim=best_params['embedding_dim'], hidden_dim=best_params['embedding_dim'], num_layers=best_params['audio_encoder_layers'])
        full_state_dict = torch.load(semantic_model_path, map_location="cpu")
        audio_encoder_state_dict = {key.replace("audio_encoder.", ""): value for key, value in full_state_dict.items() if key.startswith("audio_encoder.")}
        self.semantic_encoder.load_state_dict(audio_encoder_state_dict)
        self.projection = nn.Linear(best_params['embedding_dim'], decoder_config.d_model)

    def forward(self, input_values=None, audio_filename=""):
        print(f"\n--- Processing: {audio_filename} ---")
        device = input_values.device; self.to(device)
        acoustic_output = self.acoustic_base_model(input_values=input_values)
        features = acoustic_output.last_hidden_state
        print(f"Step 1: Acoustic Features Shape: {features.shape}"); print(f"  - Stats: mean={features.mean():.4f}, std={features.std():.4f}")
        feats_np = features[0].cpu().numpy()
        units = self.kmeans_model.predict(feats_np)
        print(f"Step 2: Acoustic Units (first 20): {units[:20]}"); print(f"  - Unique units count: {len(np.unique(units))}")
        units_tensor = torch.LongTensor(units).unsqueeze(0).to(device)
        semantic_output = self.semantic_encoder.lstm(self.semantic_encoder.embedding(units_tensor))[0]
        print(f"Step 3: Semantic Output Shape: {semantic_output.shape}"); print(f"  - Stats: mean={semantic_output.mean():.4f}, std={semantic_output.std():.4f}")
        final_hidden_states = semantic_output.mean(dim=1).unsqueeze(1)
        projected_states = self.projection(final_hidden_states)
        print(f"Step 4: Final Projected Output Shape: {projected_states.shape}"); print(f"  - Stats: mean={projected_states.mean():.4f}, std={projected_states.std():.4f}")
        return projected_states

def main():
    parser = argparse.ArgumentParser(description="Debug the SpeechToMeaningEncoder.")
    # 💥 引数を追加
    parser.add_argument("--acoustic_model_path", type=str, required=True)
    parser.add_argument("--semantic_model_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--decoder_model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--trained_model_path", type=str, required=True, help="Path to the fine-tuned model checkpoint (e.g., checkpoint-XXXX).")
    parser.add_argument("--audio_file1", type=str, required=True, help="Path to the first audio file.")
    parser.add_argument("--audio_file2", type=str, required=True, help="Path to the second audio file.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device: {device}")
    
    # 💥【修正点】モデルの読み込み方法を全面的に変更
    # 1. まず、エンコーダの骨格を finetune_translator.py と同じ方法で構築
    print("Building encoder structure...")
    encoder_config = Wav2Vec2Config.from_pretrained(args.acoustic_model_path)
    decoder_config = AutoConfig.from_pretrained(args.decoder_model_name)
    encoder = SpeechToMeaningEncoder(
        config=encoder_config,
        decoder_config=decoder_config,
        acoustic_model_path=args.acoustic_model_path,
        semantic_model_path=args.semantic_model_path,
        params_path=args.params_path,
    )
    
    # 2. チェックポイントから学習済みの重みをロード
    print(f"Loading weights from checkpoint: {args.trained_model_path}")
    # .safetensors があればそちらを優先
    safetensors_path = os.path.join(args.trained_model_path, "model.safetensors")
    bin_path = os.path.join(args.trained_model_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path, device="cpu")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Model weights not found in {args.trained_model_path}")

    # 3. エンコーダに関連する重みだけを抽出してロード
    encoder_state_dict = {
        key.replace("encoder.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device).eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.trained_model_path)
    
    # --- ここから下の推論部分は変更なし ---
    outputs = {}
    for i, audio_path in enumerate([args.audio_file1, args.audio_file2]):
        waveform, sr = torchaudio.load(audio_path)
        if sr != feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=feature_extractor.sampling_rate)
            waveform = resampler(waveform)
        inputs = feature_extractor(waveform.squeeze(0).numpy(), sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            final_output = encoder(input_values=input_values, audio_filename=os.path.basename(audio_path))
            outputs[f"audio{i+1}"] = final_output
            
    output1 = outputs["audio1"]; output2 = outputs["audio2"]
    print("\n" + "="*50); print("Final Output Comparison"); print("="*50)
    print(f"Output 1 (first 5 values): {output1.flatten()[:5].cpu().numpy()}"); print(f"Output 2 (first 5 values): {output2.flatten()[:5].cpu().numpy()}")
    are_close = torch.allclose(output1, output2, atol=1e-5)
    print(f"\nAre the two outputs nearly identical? -> {are_close}")
    if are_close: print("\n💥 Verdict: Problem confirmed. The encoder produces the same output for different inputs.")
    else: print("\n✅ Verdict: The encoder seems to be producing different outputs. The problem may lie elsewhere.")


if __name__ == "__main__":
    main()