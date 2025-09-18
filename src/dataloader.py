
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms

class Flickr8kDataset(Dataset):
    def __init__(self, data_dir, df, audio_transform=None, image_transform=None):
        self.data_dir = data_dir
        self.df = df
        self.audio_dir = os.path.join(data_dir, 'audio')
        self.image_dir = os.path.join(data_dir, 'images')
        self.audio_transform = audio_transform
        self.image_transform = image_transform

        # Use only the first caption for each image for simplicity
        self.unique_images = self.df.drop_duplicates(subset=['image'])

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):
        row = self.unique_images.iloc[idx]
        image_filename = row['image']
        
        base_image_name = os.path.splitext(image_filename)[0]
        audio_filename = f"{base_image_name}.wav"
        
        image_path = os.path.join(self.image_dir, image_filename)
        audio_path = os.path.join(self.audio_dir, audio_filename)

        try:
            # Load and transform image
            image = Image.open(image_path).convert("RGB")
            if self.image_transform:
                image = self.image_transform(image)

            # Load and transform audio
            waveform, sample_rate = torchaudio.load(audio_path)
            if self.audio_transform:
                waveform = self.audio_transform(waveform)

            # Ensure audio is mono by averaging channels if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Remove channel dimension for padding
            return waveform.squeeze(0), image

        except FileNotFoundError as e:
            print(f"Warning: File not found, skipping item. {e}")
            # Return None, and handle this in the collate_fn or DataLoader
            return None, None

def get_transforms():
    # Pre-trained ResNet expects this normalization
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # No special audio transform for now, the model's MFCC will handle it.
    # We might need resampling if sample rates differ.
    # audio_transform = torchaudio.transforms.Resample(orig_freq=..., new_freq=16000)
    audio_transform = None

    return audio_transform, image_transform

def collate_fn(batch):
    # Filter out None values from failed file loads
    batch = [b for b in batch if b[0] is not None and b[1] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])

    waveforms, images = zip(*batch)
    
    # Pad audio sequences to the same length
    # This is a simple padding strategy
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    
    images = torch.stack(images, 0)
    
    return padded_waveforms, images
