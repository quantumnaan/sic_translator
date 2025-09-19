# train_multimodal.py (最終確定版)

import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from PIL import Image
import os

from model import MultimodalAcousticModel, AcousticUnitEncoder, VisionResNetEncoder # model.pyから全てインポート
from dataloader import get_transforms

from tqdm import tqdm

# --- 定数 ---
PREPROCESSED_FILE = "data/multimodal_preprocessed.json"
NUM_EPOCHS_OPTUNA = 3 
NUM_EPOCHS_FINAL = 20
TRAIN_RATIO = 0.8
BEST_MODEL_PATH = "models/semantic_core_model/semantic_core_model.pth"

# --- データセットとデータコレータ ---
class AcousticUnitDataset(Dataset):
    def __init__(self, json_path, image_transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        units = torch.LongTensor(item['units'])
        
        try:
            image = Image.open(image_path).convert("RGB")
            if self.image_transform:
                image = self.image_transform(image)
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}, skipping.")
            return None, None

        return units, image

def collate_fn_multimodal(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])

    units_list, images_list = zip(*batch)
    padded_units = pad_sequence(units_list, batch_first=True, padding_value=0)
    images_tensor = torch.stack(images_list)
    
    return padded_units, images_tensor

# --- 損失関数 ---
def calculate_contrastive_loss(audio_emb, image_emb, temperature=0.07):
    logits = (audio_emb @ image_emb.T) / temperature
    labels = torch.arange(len(audio_emb)).to(audio_emb.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_i = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_i) / 2

# --- Optunaの目的関数 ---
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32]) # CPU/メモリに応じて調整
    temperature = trial.suggest_float("temperature", 0.01, 0.1, log=True)
    audio_encoder_layers = trial.suggest_int("audio_encoder_layers", 1, 3)

    try:
        _, image_transform = get_transforms()
        full_dataset = AcousticUnitDataset(json_path=PREPROCESSED_FILE, image_transform=image_transform)
        
        subset_indices = range(min(len(full_dataset), 200))
        subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
        
        train_size = int(TRAIN_RATIO * len(subset_dataset))
        val_size = len(subset_dataset) - train_size
        train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_multimodal)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn_multimodal)
    except FileNotFoundError:
        print(f"Error: {PREPROCESSED_FILE} not found. Please run preprocess_sound.py first.")
        return float('inf')

    # train_vq.pyで設定したクラスタ数（語彙数）に合わせる
    vocab_size = 512 
    model = MultimodalAcousticModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=audio_encoder_layers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(NUM_EPOCHS_OPTUNA):
        model.train()
        for i, (unit_sequences, images) in enumerate(train_loader):
            if unit_sequences.nelement() == 0: continue
            optimizer.zero_grad()
            audio_emb, image_emb = model(unit_sequences, images)
            loss = calculate_contrastive_loss(audio_emb, image_emb, temperature)
            loss.backward()
            optimizer.step()
            if i > 10: break

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, (unit_sequences, images) in enumerate(val_loader):
                if unit_sequences.nelement() == 0: continue
                audio_emb, image_emb = model(unit_sequences, images)
                loss = calculate_contrastive_loss(audio_emb, image_emb, temperature)
                total_val_loss += loss.item()
                if i > 10: break
        
        avg_val_loss = total_val_loss / max(1, i + 1)
        trial.report(avg_val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss

# --- 最終モデルの学習関数 ---
def train_and_save_best_model(best_params):
    print("\nTraining the best model with optimal hyperparameters...")
    lr = best_params['lr']
    embedding_dim = best_params['embedding_dim']
    batch_size = best_params['batch_size']
    temperature = best_params['temperature']
    audio_encoder_layers = best_params['audio_encoder_layers']
    
    _, image_transform = get_transforms()
    full_dataset = AcousticUnitDataset(json_path=PREPROCESSED_FILE, image_transform=image_transform)
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_multimodal)

    vocab_size = 512
    model = MultimodalAcousticModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=audio_encoder_layers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(NUM_EPOCHS_FINAL):
        total_loss = 0
        for unit_sequences, images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS_FINAL}"):
            if unit_sequences.nelement() == 0: continue
            optimizer.zero_grad()
            audio_emb, image_emb = model(unit_sequences, images)
            loss = calculate_contrastive_loss(audio_emb, image_emb, temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS_FINAL}, Average Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f"\nBest model saved to {BEST_MODEL_PATH}")


# --- メイン実行ブロック ---
if __name__ == "__main__":
    study = optuna.create_study(
        study_name="multimodal_translator_study",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
        direction="minimize"
    )
    
    study.optimize(objective, n_trials=20)
    
    print("\n--- Optuna Optimization Finished ---")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Validation Loss: {trial.value}")
    print("  Best Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 最適なパラメータで最終モデルを学習し保存
    train_and_save_best_model(study.best_params) # ← 本格的な学習を行う場合はこの行のコメントを外す
    # print("\nTo train the final model with these parameters, uncomment the 'train_and_save_best_model' line in the script.")