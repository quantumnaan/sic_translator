import argparse
import os
from typing import Dict, List, Optional, Union

import torch
from datasets import load_dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Trainer,
    TrainingArguments,
)

# --- デバッグ用のカスタムTrainer ---


class DebuggingTrainer(Trainer):
    def training_step(self, model, inputs, *args, **kwargs):
        print("--- Inside training_step ---")
        # 入力テンソルの形状を確認
        print("Inputs keys:", inputs.keys())
        print("Inputs shape:", inputs['input_values'].shape)

        # モデルを評価モードにして、まず損失なしで何が返るか確認
        model.eval()
        with torch.no_grad():
            outputs_eval = model(**inputs)
        print("--- Outputs in eval mode ---")
        print(outputs_eval.keys())

        # モデルを訓練モードに戻す
        model.train()
        # 訓練モードでフォワードパスを実行
        outputs_train = model(**inputs)
        print("--- Outputs in train mode ---")
        print(outputs_train.keys())

        # デバッグ情報を表示したら、ここで処理を停止
        raise RuntimeError("Debugging complete. Stopping training.")


def main(args):
    """
    教師なしファインチューニングのメイン関数
    """
    # 1. データセットの読み込みと前処理
    # ---------------------------------
    print("--- Loading and Preprocessing Data ---")
    raw_datasets = load_dataset(
        "audiofolder", data_dir=args.data_dir, split="train")

    raw_datasets = raw_datasets.select(range(5))
    print(f"--- Dataset downsampled to 5 for testing ---")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.base_model)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            padding=True,
            max_length=int(feature_extractor.sampling_rate * 10.0),
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
    )

    print(f"Dataset prepared. Number of examples: {len(processed_datasets)}")

    # 2. モデルと学習設定の準備
    # ---------------------------
    print("--- Preparing Model and Training Arguments ---")
    model = Wav2Vec2ForPreTraining.from_pretrained(args.base_model)

    # --- モデル設定のデバッグ出力 ---
    print("--- Model Config ---")
    print(model.config)
    # --------------------------

    if torch.cuda.is_available():
        model.to("cuda")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
        logging_steps=1,
        save_steps=500,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="wandb" if "WANDB_API_KEY" in os.environ else "none",
        push_to_hub=False,
    )

    # 3. トレーナーの初期化と学習の開始
    # ---------------------------------
    trainer = DebuggingTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets,
    )

    print("--- Starting Training (in debug mode) ---")
    try:
        trainer.train()
    except RuntimeError as e:
        print(f"Caught expected exception: {e}")

    print("--- Debugging run finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wav2Vec2の教師なしファインチューニングスクリプト")

    # use a smaller default model to reduce memory and compute requirements
    parser.add_argument(
        "--base_model", type=str, default="facebook/wav2vec2-base")
    parser.add_argument(
        "--data_dir", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5)
    # lower defaults to reduce memory pressure for small machines
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=2)
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1)

    args = parser.parse_args()
    main(args)
