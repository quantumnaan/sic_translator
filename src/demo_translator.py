# src/demo_translator.py (Final Manual Fix Version)

import argparse
import torch
import torchaudio
import joblib
import json
import os
from safetensors.torch import load_file
from transformers import (
    Wav2Vec2FeatureExtractor,
    AutoTokenizer,
    EncoderDecoderModel,
    Wav2Vec2Config,
    AutoConfig,
    AutoModelForCausalLM
)
from finetune_translator import SpeechToMeaningEncoder

def main():
    parser = argparse.ArgumentParser(
        description="Translate a single audio file using the final fine-tuned model."
    )
    # ... (引数の定義は変更なし) ...
    parser.add_argument("--acoustic_model_path", type=str, required=True)
    parser.add_argument("--semantic_model_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--decoder_model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--audio_file", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. モデルの「骨格」を構築
    print("Building model structure...")
    encoder_config = Wav2Vec2Config.from_pretrained(args.acoustic_model_path)
    decoder_config = AutoConfig.from_pretrained(args.decoder_model_name)
    encoder = SpeechToMeaningEncoder(
        config=encoder_config, decoder_config=decoder_config,
        acoustic_model_path=args.acoustic_model_path, semantic_model_path=args.semantic_model_path,
        params_path=args.params_path,
    )
    decoder = AutoModelForCausalLM.from_pretrained(args.decoder_model_name)
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    # 2. チェックポイントから重みをロード
    print(f"Loading weights from checkpoint: {args.model_path}")
    weights_path = os.path.join(args.model_path, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(args.model_path, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found in {args.model_path}")

    if weights_path.endswith(".safetensors"):
        state_dict = load_file(weights_path, device="cpu")
    else:
        state_dict = torch.load(weights_path, map_location="cpu")

    # 💥【最終修正】 'decoder.lm_head.weight' が存在しない問題を手術的に解決
    # 埋め込み層の重みを、欠けているlm_headの重みとして手動でコピー（重みを共有）する
    print("Manually correcting state_dict for tied weights...")
    embedding_weight_key = "decoder.model.decoder.embed_tokens.weight"
    if embedding_weight_key in state_dict:
        state_dict["decoder.lm_head.weight"] = state_dict[embedding_weight_key]
    else:
        # 念のためのフォールバック
        raise KeyError(f"Could not find the source for tied weights: '{embedding_weight_key}' in the checkpoint.")

    # 3. 修正したstate_dictをモデルに適用
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("Model loaded successfully!")
    
    # 4. Feature Extractor と Tokenizer をロード
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # ... (ここから下の音声処理と推論の部分は変更なし) ...
    print(f"Loading and processing audio file: '{args.audio_file}'...")
    waveform, sample_rate = torchaudio.load(args.audio_file)
    if sample_rate != feature_extractor.sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=feature_extractor.sampling_rate)
        waveform = resampler(waveform)
    inputs = feature_extractor(waveform.squeeze(0).numpy(), sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    print("Translating...")
    with torch.no_grad():
        predicted_ids = model.generate(input_values)

    transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print("\n" + "="*50); print("Translation Result"); print("="*50)
    print(f"Input Audio: {args.audio_file}")
    print(f"Translated Text: {transcription}")
    print("="*50)

if __name__ == "__main__":
    main()