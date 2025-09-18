
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd

from model import MultimodalModel
from dataloader import Flickr8kDataset, get_transforms, collate_fn

# --- CPUに関する警告 ---
# 以下の学習プロセスはCPUでは非常に長い時間がかかります。
# まずはデータセットのサブセット（例: 100件）でコードが
# 正常に動作することを確認してから、本番の学習を行ってください。
# --------------------

from tqdm import tqdm

# --- 定数 ---
DATA_DIR = "data"
CAPTIONS_FILE = f"{DATA_DIR}/captions.txt"
NUM_EPOCHS_OPTUNA = 3 # Optunaでの動作確認のためエポック数を少なく設定
NUM_EPOCHS_FINAL = 10 # 最終モデル学習のためエポック数を設定
TRAIN_RATIO = 0.8
BEST_MODEL_PATH = "best_model.pth"

def calculate_contrastive_loss(audio_emb, image_emb, temperature=0.07):
    # InfoNCE損失（CLIPで使われる損失関数）を計算
    logits = (audio_emb @ image_emb.T) / temperature
    labels = torch.arange(len(audio_emb)).to(audio_emb.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_i = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_i) / 2

def objective(trial):
    # 1. ハイパーパラメータの定義
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256]) # 512はCPUには大きすぎる可能性
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    temperature = trial.suggest_float("temperature", 0.01, 0.1, log=True)

    # 2. モデル、データローダー、オプティマイザのセットアップ
    try:
        df = pd.read_csv(CAPTIONS_FILE)
    except FileNotFoundError:
        print(f"Error: {CAPTIONS_FILE} not found. Please run prepare_audio_data.py first and ensure the dataset is set up.")
        # Optunaに失敗を伝えるために大きな値を返す
        return float('inf')

    # --- データセットのサブセットでテスト ---
    df = df.head(100) # 動作確認用にデータセットを100件に絞る
    # -------------------------------------

    audio_transform, image_transform = get_transforms()
    dataset = Flickr8kDataset(data_dir=DATA_DIR, df=df, image_transform=image_transform)

    # データセットを訓練用と検証用に分割
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = MultimodalModel(audio_model_name=None, vision_model_name=None, embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3. 学習ループ
    for epoch in range(NUM_EPOCHS_OPTUNA):
        model.train()
        for i, (waveforms, images) in enumerate(train_loader):
            if waveforms.nelement() == 0: continue # 空のバッチをスキップ

            optimizer.zero_grad()
            
            audio_emb, image_emb = model(waveforms, images)
            
            loss = calculate_contrastive_loss(audio_emb, image_emb, temperature)
            loss.backward()
            optimizer.step()

            # 動作確認のため、最初の数バッチでループを抜ける
            if i > 5:
                break

        # 4. バリデーションと結果の報告
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, (waveforms, images) in enumerate(val_loader):
                if waveforms.nelement() == 0: continue

                audio_emb, image_emb = model(waveforms, images)
                loss = calculate_contrastive_loss(audio_emb, image_emb, temperature)
                total_val_loss += loss.item()
                if i > 5:
                    break
        
        avg_val_loss = total_val_loss / (len(val_loader) if len(val_loader) > 0 else 1)
        trial.report(avg_val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss

def train_and_save_best_model(best_params):
    print("\nTraining the best model with optimal hyperparameters...")

    # 1. ベストなハイパーパラメータでセットアップ
    lr = best_params['lr']
    embedding_dim = best_params['embedding_dim']
    batch_size = best_params['batch_size']
    temperature = best_params['temperature']

    # 2. データローダーのセットアップ
    df = pd.read_csv(CAPTIONS_FILE)
    df = df.head(100) # 動作確認用にデータセットを100件に絞る
    audio_transform, image_transform = get_transforms()
    
    # 全データを訓練に使用
    full_dataset = Flickr8kDataset(data_dir=DATA_DIR, df=df, image_transform=image_transform)
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 3. モデルのセットアップ
    model = MultimodalModel(audio_model_name=None, vision_model_name=None, embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. 学習ループ
    model.train()
    for epoch in range(NUM_EPOCHS_FINAL):
        total_loss = 0
        for waveforms, images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS_FINAL}"):
            if waveforms.nelement() == 0: continue

            optimizer.zero_grad()
            audio_emb, image_emb = model(waveforms, images)
            loss = calculate_contrastive_loss(audio_emb, image_emb, temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS_FINAL}, Average Loss: {avg_loss:.4f}")

    # 5. モデルの保存
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f"\nBest model saved to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="sic_translator_study", 
        storage="sqlite:///optuna_study.db", 
        load_if_exists=True, 
        direction="minimize"
    )
    # 試行回数を少なく設定して動作確認
    study.optimize(objective, n_trials=10)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 最適なパラメータでモデルを学習し保存
    train_and_save_best_model(study.best_params)
