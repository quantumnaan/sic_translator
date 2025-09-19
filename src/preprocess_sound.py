"""

目的:
マルチモーダルデータセットの各音声ファイルを、フェーズAで学習した
音響単位発見モデルに通し、「音響単位の系列」に変換する。
結果を、対応する画像ファイルパスと共にJSONファイルとして保存する。
このスクリプトは、フェーズBの本格的な学習の前に一度だけ実行する。

実行例:
python preprocess_for_phase_b.py \
  --acoustic_model_path "models/acoustic_unit_model/" \
  --multimodal_data_dir "data/multimodal_pairs/" \
  --output_file "data/multimodal_preprocessed.json"
"""

import argparse
import os
import glob
import json
from tqdm import tqdm

import torch
import torchaudio
from transformers import Wav2Vec2Model
import joblib
import numpy as np

def get_args():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(description="Preprocess audio files into acoustic unit sequences for Phase B.")
    parser.add_argument("--acoustic_model_path", type=str, required=True, help="Path to the directory containing the trained acoustic unit model (base model + k-means model).")
    parser.add_argument("--multimodal_data_dir", type=str, required=True, help="Directory containing 'audio/' and 'images/' subdirectories.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate for audio.")
    
    return parser.parse_args()

def load_audio(file_path, target_sample_rate):
    """音声ファイルを読み込み、リサンプリングしてモノラルに変換する"""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

@torch.no_grad()
def main():
    args = get_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. フェーズAで学習したモデル（ベースモデルとK-Means）をロード
    print(f"Loading acoustic unit model from {args.acoustic_model_path}...")
    try:
        base_model = Wav2Vec2Model.from_pretrained(args.acoustic_model_path).to(device)
        base_model.eval()
        
        kmeans_path = os.path.join(args.acoustic_model_path, "kmeans_codebook.joblib")
        kmeans_model = joblib.load(kmeans_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure the directory contains both the Hugging Face model files and the 'kmeans_codebook.joblib' file.")
        return

    # 2. 対応する音声と画像のペアを探す
    audio_files = glob.glob(os.path.join(args.multimodal_data_dir, "audio", "*.wav"))
    image_dir = os.path.join(args.multimodal_data_dir, "images")
    
    data_pairs = []
    for audio_path in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        # .jpg, .pngなど複数の拡張子に対応
        possible_image_paths = glob.glob(os.path.join(image_dir, f"{base_name}.*"))
        if possible_image_paths:
            # 拡張子が画像形式であるかを簡易チェック
            img_ext = os.path.splitext(possible_image_paths[0])[1].lower()
            if img_ext in ['.jpg', '.jpeg', '.png']:
                data_pairs.append({"audio": audio_path, "image": possible_image_paths[0]})

    print(f"Found {len(data_pairs)} corresponding audio-image pairs.")

    # 3. 各音声ファイルを処理し、音響単位系列に変換
    results = []
    for pair in tqdm(data_pairs, desc="Preprocessing audio files"):
        try:
            waveform = load_audio(pair["audio"], args.sample_rate).to(device)
            
            # ベースモデルで特徴量を抽出
            features = base_model(waveform).last_hidden_state.squeeze(0) # (time, hidden_dim)
            features_np = features.cpu().numpy()

            if features_np.shape[0] == 0:
                print(f"Warning: Zero features for {pair['audio']}, skipping.")
                continue

            # K-Meansモデルで各特徴ベクトルをクラスタID（音響単位）に変換
            acoustic_units = kmeans_model.predict(features_np)
            
            # 結果を保存
            results.append({
                "image_path": pair["image"],
                "units": acoustic_units.tolist() # JSONで保存できるようPythonのリストに変換
            })
        except Exception as e:
            print(f"Warning: Failed to process {pair['audio']}. Error: {e}")

    # 4. 結果をJSONファイルとして保存
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("\n--- Preprocessing for Phase B Complete ---")
    print(f"Successfully processed {len(results)} pairs.")
    print(f"Output saved to: {args.output_file}")
    print("This file is now ready to be used as input for 'train_multimodal.py'.")

if __name__ == "__main__":
    main()