# generate_embeddings.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from model import MultimodalModel
from dataloader import Flickr8kDataset, get_transforms, collate_fn

# --- 定数 ---
DATA_DIR = "data"
CAPTIONS_FILE = f"{DATA_DIR}/captions.txt"
BEST_MODEL_PATH = "best_model.pth"
# これは保存されたモデルと一致する必要があります
# 前回の実行から、最適なembedding_dimは256でした
EMBEDDING_DIM = 256 
BATCH_SIZE = 32 # メモリに応じて調整可能

IMAGE_EMBEDDINGS_PATH = "image_embeddings.pt"
AUDIO_EMBEDDINGS_PATH = "audio_embeddings.pt"
IMAGE_FILENAMES_PATH = "image_filenames.json"

def generate_embeddings():
    print("Generating embeddings from the dataset...")

    # 1. データローダーのセットアップ
    df = pd.read_csv(CAPTIONS_FILE)
    df = df.head(100) # 学習時と同じサブセットを使用
    audio_transform, image_transform = get_transforms()
    
    dataset = Flickr8kDataset(data_dir=DATA_DIR, df=df, image_transform=image_transform)
    # shuffle=Falseを使い、ファイル名とembeddingの順序を維持する
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 2. モデルのセットアップ
    model = MultimodalModel(audio_model_name=None, vision_model_name=None, embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval() # モデルを評価モードに設定

    # 3. 特徴量ベクトルの生成
    all_audio_embeddings = []
    all_image_embeddings = []
    
    with torch.no_grad():
        for waveforms, images in tqdm(data_loader, desc="Generating Embeddings"):
            if waveforms.nelement() == 0: continue

            audio_emb, image_emb = model(waveforms, images)
            
            all_audio_embeddings.append(audio_emb)
            all_image_embeddings.append(image_emb)

    # 全てのバッチを結合
    all_audio_embeddings = torch.cat(all_audio_embeddings, dim=0)
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)

    # 4. ファイル名のリストを取得
    image_filenames = dataset.unique_images['image'].tolist()

    # データローダーがファイル欠損などでアイテムをスキップした場合、
    # embeddingとファイル名の数が一致しない可能性があるため、長さを揃える
    if len(all_image_embeddings) != len(image_filenames):
        print(f"Warning: Mismatch in length between embeddings ({len(all_image_embeddings)}) and filenames ({len(image_filenames)}). Truncating to the shorter length.")
        min_len = min(len(all_image_embeddings), len(image_filenames))
        all_image_embeddings = all_image_embeddings[:min_len]
        all_audio_embeddings = all_audio_embeddings[:min_len]
        image_filenames = image_filenames[:min_len]

    # 5. ベクトルとファイル名を保存
    torch.save(all_image_embeddings, IMAGE_EMBEDDINGS_PATH)
    torch.save(all_audio_embeddings, AUDIO_EMBEDDINGS_PATH)
    with open(IMAGE_FILENAMES_PATH, 'w') as f:
        json.dump(image_filenames, f)

    print(f"Embeddings and filenames saved successfully.")
    print(f"Image embeddings shape: {all_image_embeddings.shape}")
    print(f"Audio embeddings shape: {all_audio_embeddings.shape}")
    print(f"Number of filenames: {len(image_filenames)}")


if __name__ == "__main__":
    generate_embeddings()
