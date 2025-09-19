# train_vq.py

"""
train_vq.py

目的:
ファインチューニング済みのWav2Vec2ベースモデルから特徴量を抽出し、
K-Meansクラスタリングを用いてVector Quantization (VQ) レイヤーの
コードブックを学習する。最終的に、ベースモデルとVQレイヤーを
組み合わせた「音響単位発見モデル」を保存する。

実行例:
python train_vq.py \
  --base_model_path "models/aud_base_model/" \
  --data_dir "data/target_lang_unlabeled/" \
  --output_dir "models/acoustic_unit_model/" \
  --num_clusters 512 \
  --sample_rate 16000 \
  --max_files_for_kmeans 1000
"""

import argparse
import os
import glob
from tqdm import tqdm

import torch
import torchaudio
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor # 💥 FeatureExtractorをインポート


def get_args():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(description="Train VQ layer using K-Means on extracted features.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the fine-tuned base model directory.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing unlabeled audio files (.wav).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final acoustic unit model.")
    parser.add_argument("--num_clusters", type=int, default=512, help="Number of clusters for K-Means (codebook size).")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate for audio.")
    parser.add_argument("--max_files_for_kmeans", type=int, default=1000, help="Maximum number of audio files to use for K-Means training to save memory/time.")
    
    return parser.parse_args()

def load_audio(file_path, target_sample_rate):
    """音声ファイルを読み込み、リサンプリングする"""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0)

@torch.no_grad()
def extract_features(model, data_dir, max_files, sample_rate, device):
    """
    音声ファイルから特徴量を抽出し、K-Means学習用のデータセットを作成する。
    メモリを節約するため、各ファイルからランダムに特徴ベクトルをサンプリングする。
    """
    model.to(device)
    model.eval()

    filepaths = glob.glob(os.path.join(data_dir, "*.wav"))
    if len(filepaths) > max_files:
        filepaths = np.random.choice(filepaths, max_files, replace=False)

    all_features = []
    print(f"Extracting features from {len(filepaths)} audio files...")
    for path in tqdm(filepaths):
        try:
            waveform = load_audio(path, sample_rate)
            # 長すぎる音声はチャンクに分割（ここでは単純化のため、先頭部分のみ使用）
            max_len = sample_rate * 30 # 30秒
            if len(waveform) > max_len:
                waveform = waveform[:max_len]
            
            waveform = waveform.to(device)
            
            # 特徴量を抽出
            features = model(waveform.unsqueeze(0)).last_hidden_state.squeeze(0) # (time, hidden_dim)
            
            # 全ての特徴量を使うと巨大になるため、各ファイルから最大200個をランダムサンプリング
            if features.shape[0] > 200:
                indices = np.random.choice(features.shape[0], 200, replace=False)
                features = features[indices]

            all_features.append(features.cpu().numpy())

        except Exception as e:
            print(f"Warning: Skipping file {path} due to error: {e}")

    return np.concatenate(all_features, axis=0)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. ファインチューニング済みのベースモデルをロード
    print(f"Loading base model and feature extractor from {args.base_model_path}...")
    base_model = Wav2Vec2Model.from_pretrained(args.base_model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model_path) # 💥【追加】
    
    # 2. 特徴量を抽出
    feature_vectors = extract_features(
        model=base_model,
        data_dir=args.data_dir,
        max_files=args.max_files_for_kmeans,
        sample_rate=args.sample_rate,
        device=device
    )
    print(f"Extracted a total of {feature_vectors.shape[0]} feature vectors for clustering.")

    # 3. K-Meansでコードブックを学習
    print(f"Training K-Means with {args.num_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.num_clusters,
        random_state=0,
        batch_size=256,
        verbose=1,
        max_iter=100,
    )
    kmeans.fit(feature_vectors)
    
    # K-Meansモデル（コードブック情報を含む）を保存
    kmeans_path = os.path.join(args.output_dir, "kmeans_codebook.joblib")
    joblib.dump(kmeans, kmeans_path)
    print(f"K-Means model saved to {kmeans_path}")

    # 4. ベースモデルとコードブックを統合して最終モデルを保存
    # transformersはVQレイヤーを直接サポートしていないため、ここでは
    # ベースモデルとコードブック（K-Meansのクラスタ中心）を別々に保存する。
    # 推論時にこれらを組み合わせて使用する。
    base_model.save_pretrained(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir) # 💥【追加】
    print(f"Final base model saved to {args.output_dir}")
    print("\n--- Training VQ Layer Complete ---")
    print(f"The final acoustic unit model consists of two parts:")
    print(f"1. The base feature extractor: saved in '{args.output_dir}'")
    print(f"2. The codebook (from K-Means): saved as '{kmeans_path}'")
    print("In the next phase (Phase B), load both to convert audio into acoustic unit sequences.")

if __name__ == "__main__":
    main()