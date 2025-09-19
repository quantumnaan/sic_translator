"""

目的:
フェーズAとフェーズBで学習したモデルを用いて、入力された音声クリップの
意味を理解し、それに最も近い画像を画像ギャラリーから検索して表示するデモ。

実行例:
python src/demo_aud2img.py \
    --acoustic_model_path models/acoustic_unit_model/ \
    --model_dir models/semantic_core_model/ \
    --image_gallery_dir data/images/ \
    --query_audio_file data/audio/10815824_2997e03d76.wav \
    --top_k 5
"""

import argparse
import os
import glob
from tqdm import tqdm

import torch
import torchaudio
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
from transformers import Wav2Vec2Model

# 以前のスクリプトから必要なクラスや関数をインポート
from model import MultimodalAcousticModel 
from dataloader import get_transforms

def get_args():
    parser = argparse.ArgumentParser(description="Demo for audio-to-image retrieval using Phase B model.")
    parser.add_argument("--acoustic_model_path", type=str, required=True, help="...")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing the semantic model and best_params.json.")
    parser.add_argument("--image_gallery_dir", type=str, required=True, help="...")
    parser.add_argument("--query_audio_file", type=str, required=True, help="...")
    parser.add_argument("--top_k", type=int, default=5, help="...")
    parser.add_argument("--sample_rate", type=int, default=16000, help="...")
    parser.add_argument("--vocab_size", type=int, default=512, help="...")
    return parser.parse_args()

@torch.no_grad()
def create_or_load_image_index(model, image_dir, index_path, device):
    """画像ギャラリーの全画像の意味ベクトルを計算し、インデックスとして保存・ロードする"""
    if os.path.exists(index_path):
        print(f"Loading existing image index from {index_path}...")
        return torch.load(index_path)

    print(f"Creating new image index, saving to {index_path}...")
    model.to(device)
    model.eval()
    
    _, image_transform = get_transforms()
    image_paths = glob.glob(os.path.join(image_dir, "*.*"))
    image_embeddings = []
    valid_image_paths = []

    for img_path in tqdm(image_paths, desc="Indexing images"):
        try:
            image = Image.open(img_path).convert("RGB")
            image = image_transform(image).unsqueeze(0).to(device)
            
            # 画像エンコーダで意味ベクトルを計算
            img_emb = model.vision_encoder(image)
            image_embeddings.append(img_emb.cpu())
            valid_image_paths.append(img_path)
        except Exception as e:
            print(f"Warning: Could not process image {img_path}. Error: {e}")

    if not image_embeddings:
        raise ValueError("No valid images found to create an index.")
        
    index = {
        "paths": valid_image_paths,
        "embeddings": torch.cat(image_embeddings, dim=0)
    }
    torch.save(index, index_path)
    return index

@torch.no_grad()
def get_audio_embedding(audio_path, acoustic_base_model, kmeans_model, semantic_model, sample_rate, device):
    """単一の音声ファイルから最終的な意味ベクトルを計算する"""
    acoustic_base_model.to(device)
    semantic_model.to(device)
    acoustic_base_model.eval()
    semantic_model.eval()

    # 1. 音声ファイルをロード
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.to(device)

    # 2. 音響単位系列に変換 (フェーズAのモデルを使用)
    features = acoustic_base_model(waveform).last_hidden_state.squeeze(0)
    units = kmeans_model.predict(features.cpu().numpy())
    units_tensor = torch.LongTensor(units).unsqueeze(0).to(device)

    # 3. 意味ベクトルに変換 (フェーズBのモデルを使用)
    audio_embedding = semantic_model.audio_encoder(units_tensor)
    return audio_embedding

def display_results(query_path, result_paths, result_scores):
    """検索結果の画像を表示する"""
    plt.figure(figsize=(20, 5))
    
    # 最初のsubplotはクエリ音声を示す（今回はファイル名のみ）
    plt.subplot(1, len(result_paths) + 1, 1)
    plt.text(0.5, 0.5, f"Query:\n{os.path.basename(query_path)}", ha='center', va='center', fontsize=12)
    plt.axis('off')

    for i, (path, score) in enumerate(zip(result_paths, result_scores)):
        img = Image.open(path).convert("RGB")
        plt.subplot(1, len(result_paths) + 1, i + 2)
        plt.imshow(img)
        plt.title(f"Score: {score:.4f}")
        plt.axis('off')
    
    plt.suptitle("Audio-to-Image Retrieval Results", fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 💥【ここから修正】JSONファイルからパラメータを読み込む
    params_path = os.path.join(args.model_dir, "best_params.json")
    semantic_model_path = os.path.join(args.model_dir, "semantic_core_model.pth")
    
    try:
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded best hyperparameters from {params_path}")
    except FileNotFoundError:
        print(f"Error: {params_path} not found. Please run train_multimodal.py first.")
        return

    # 1. 学習済みモデルをすべてロード
    # フェーズAのモデル
    acoustic_base_model = Wav2Vec2Model.from_pretrained(args.acoustic_model_path)
    kmeans_path = os.path.join(args.acoustic_model_path, "kmeans_codebook.joblib")
    kmeans_model = joblib.load(kmeans_path)
    
    # フェーズBのモデル

    # フェーズBのモデルを、読み込んだパラメータで初期化
    semantic_model = MultimodalAcousticModel(
        vocab_size=args.vocab_size, 
        embedding_dim=best_params['embedding_dim'], 
        num_layers=best_params['audio_encoder_layers']
    )
    semantic_model.load_state_dict(torch.load(semantic_model_path, map_location=device))

    # 2. 画像インデックスを作成またはロード
    index_path = os.path.join(os.path.dirname(args.model_dir), "image_index.pt")
    image_index = create_or_load_image_index(semantic_model, args.image_gallery_dir, index_path, device)
    
    # 3. クエリ音声の意味ベクトルを計算
    print(f"Processing query audio: {args.query_audio_file}...")
    query_audio_emb = get_audio_embedding(
        args.query_audio_file, 
        acoustic_base_model, 
        kmeans_model, 
        semantic_model, 
        args.sample_rate, 
        device
    )

    # 4. 検索の実行（コサイン類似度）
    print("Searching for similar images...")
    image_embeddings = image_index["embeddings"].to(device)
    similarities = F.cosine_similarity(query_audio_emb, image_embeddings)
    
    # 類似度が高い順にトップKを取得
    top_k_scores, top_k_indices = torch.topk(similarities, args.top_k)

    result_paths = [image_index["paths"][i] for i in top_k_indices.cpu().numpy()]
    result_scores = top_k_scores.cpu().numpy()

    # 5. 結果の表示
    print("\n--- Top K Results ---")
    for i, (path, score) in enumerate(zip(result_paths, result_scores)):
        print(f"{i+1}: {os.path.basename(path)} (Similarity: {score:.4f})")
    
    display_results(args.query_audio_file, result_paths, result_scores)

if __name__ == "__main__":
    main()