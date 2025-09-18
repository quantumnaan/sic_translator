# pretrain_wav2vec2_trainer.py
import argparse
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2ForPreTraining,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
)

# helpers from transformers (internal but used by examples)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices

def get_args():
    parser = argparse.ArgumentParser(description="Unsupervised pre-training of Wav2Vec2 (using Trainer).")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_audio_length_seconds", type=int, default=15)
    parser.add_argument("--sample_rate", type=int, default=16000)
    return parser.parse_args()

class AudioDataset(Dataset):
    def __init__(self, filepaths, max_length_seconds, sample_rate):
        self.filepaths = filepaths
        self.max_length_samples = max_length_seconds * sample_rate
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        try:
            waveform, sr = torchaudio.load(filepath)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if waveform.shape[1] > self.max_length_samples:
                start_idx = torch.randint(0, waveform.shape[1] - self.max_length_samples + 1, (1,)).item()
                waveform = waveform[:, start_idx : start_idx + self.max_length_samples]
            # return numpy 1D array (feature_extractor.pad expects list/np arrays)
            return {"input_values": waveform.squeeze(0).numpy()}
        except Exception as e:
            print(f"Warning: Error loading {filepath}, skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))

@dataclass
class SimpleDataCollator:
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[float], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # feature_extractor.pad returns dict with "input_values": tensor (batch, time)
        return batch


class Wav2Vec2PreTrainingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        inputs["input_values"]: torch.Tensor, shape (B, T)
        We must:
          - compute feature-level sequence length after convs,
          - build mask_time_indices with _compute_mask_indices,
          - call _sample_negative_indices with features_shape=(B, seq_len),
            which returns an array of shape (B, seq_len, num_negatives),
          - convert masks / indices to torch tensors on model device,
          - call model(..., mask_time_indices=..., sampled_negative_indices=...).
        """
        input_values = inputs["input_values"]
        device = next(model.parameters()).device
        input_values = input_values.to(device).float()

        batch_size, raw_seq_len = input_values.shape

        # 1) compute feature-level seq len after feature-extractor convs
        feat_seq_len = model._get_feat_extract_output_lengths(raw_seq_len)
        feat_seq_len = int(feat_seq_len) if not isinstance(feat_seq_len, torch.Tensor) else int(feat_seq_len.item())

        # 2) compute mask_time_indices (numpy bool array) for shape (B, feat_seq_len)
        mask_time_indices = _compute_mask_indices(
            (batch_size, feat_seq_len),
            mask_prob=getattr(model.config, "mask_time_prob", 0.05),
            mask_length=getattr(model.config, "mask_time_length", 10),
            min_masks=getattr(model.config, "mask_time_min_masks", 0),
        )  # -> numpy array shape (B, feat_seq_len), dtype=bool

        # 3) sample negative indices — PASS features_shape=(batch_size, feat_seq_len)
        sampled_negative_indices = _sample_negative_indices(
            features_shape=(batch_size, feat_seq_len),             # <- 2-tuple !
            num_negatives=getattr(model.config, "num_negatives", 100),
            mask_time_indices=mask_time_indices,
        )  # -> numpy array shape (B, feat_seq_len, num_negatives), dtype=int

        # 4) convert to torch tensors on correct device / dtype
        mask_time_indices = torch.tensor(mask_time_indices, dtype=torch.bool, device=device)
        sampled_negative_indices = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        # 5) forward (model will compute loss when mask + negatives are provided)
        outputs = model(
            input_values=input_values,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss



def main():
    args = get_args()

    print(f"Loading base model '{args.base_model}'...")
    model = Wav2Vec2ForPreTraining.from_pretrained(args.base_model)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model)

    print(f"Loading audio files from '{args.data_dir}'...")
    filepaths = glob.glob(os.path.join(args.data_dir, "*.wav"))
    if not filepaths:
        raise ValueError(f"No .wav files found in {args.data_dir}")

    train_dataset = AudioDataset(filepaths, args.max_audio_length_seconds, args.sample_rate)
    data_collator = SimpleDataCollator(feature_extractor=feature_extractor, padding=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        remove_unused_columns=False,
        label_names=[],
        warmup_steps=500,
    )

    trainer = Wav2Vec2PreTrainingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\n--- Starting Unsupervised Pre-Training ---")
    trainer.train()
    print("\n--- Pre-Training Complete ---")

    # Save base wav2vec2 encoder (w/o VQ head)
    model.wav2vec2.save_pretrained(args.output_dir)
    print(f"Fine-tuned base model saved to '{args.output_dir}'")

if __name__ == "__main__":
    main()
