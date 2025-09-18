# inference.py
import torch
import torch.nn.functional as F
import pandas as pd
import torchaudio
import json
import argparse

from model import MultimodalModel

# --- 定数 ---
DATA_DIR = "data"
CAPTIONS_FILE = f"{DATA_DIR}/captions.txt"
BEST_MODEL_PATH = "best_model.pth"
EMBEDDING_DIM = 256 # 保存されたモデルと一致させる必要があります
IMAGE_EMBEDDINGS_PATH = "image_embeddings.pt"
IMAGE_FILENAMES_PATH = "image_filenames.json"

def find_best_match(audio_path, model, image_embeddings, image_filenames, captions_df):
    """
    与えられた音声ファイルに最もマッチする画像を検索します。
    """
    print(f"Finding best match for audio: {audio_path}")

    # 1. 音声ファイルを読み込んで前処理
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        # モノラルに変換し、(1, time)の形状を確実にする
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # これで波形は(1, time)となり、バッチサイズ1の入力として正しい
    except Exception as e:
        print(f"Error loading or processing audio file: {e}")
        return

    # 2. 音声の埋め込みベクトルを計算
    model.eval()
    with torch.no_grad():
        audio_embedding, _ = model(audio_inputs=waveform)

    # 3. コサイン類似度を計算
    audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    
    similarities = torch.matmul(audio_embedding, image_embeddings.T).squeeze(0)

    # 4. 最も類似度が高い画像を見つける
    best_match_index = torch.argmax(similarities).item()
    best_match_similarity = similarities[best_match_index].item()
    best_match_filename = image_filenames[best_match_index]

    # 5. 対応するキャプションを検索
    caption = captions_df[captions_df['image'] == best_match_filename]['caption'].iloc[0]

    print("\n--- Inference Result ---")
    print(f"Best match image: {best_match_filename}")
    print(f"Cosine Similarity: {best_match_similarity:.4f}")
    print(f"Found Caption: {caption}")


def main():
    parser = argparse.ArgumentParser(description="Translate audio to image caption.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file (.wav).")
    args = parser.parse_args()

    # 1. 必要なファイルをロード
    print("Loading model and embeddings...")
    try:
        model = MultimodalModel(audio_model_name=None, vision_model_name=None, embedding_dim=EMBEDDING_DIM)
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        
        image_embeddings = torch.load(IMAGE_EMBEDDINGS_PATH)
        
        with open(IMAGE_FILENAMES_PATH, 'r') as f:
            image_filenames = json.load(f)
            
        captions_df = pd.read_csv(CAPTIONS_FILE)
    except FileNotFoundError as e:
        print(f"Error loading a required file: {e}")
        print("Please make sure you have run train.py and generate_embeddings.py first.")
        return
    
    print("Loading complete.")

    # 2. マッチングを実行
    find_best_match(args.audio_path, model, image_embeddings, image_filenames, captions_df)


if __name__ == "__main__":
    main()
