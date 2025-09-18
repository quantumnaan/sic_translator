# model.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, ViTModel
import torchaudio
from torchvision import models

class MultimodalModel(nn.Module):
    def __init__(self, audio_model_name, vision_model_name, embedding_dim):
        super(MultimodalModel, self).__init__()
        # CPUでの実行を考慮し、モデルサイズに注意
        #self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        #self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        
        # CPU向け簡易版: MFCC + LSTM
        self.audio_encoder = AudioLSTMEncoder(input_size=40, hidden_size=embedding_dim)
        # CPU向け簡易版: 事前学習済みResNet
        self.vision_encoder = VisionResNetEncoder(embedding_dim=embedding_dim)

        # 最終的な特徴量を共通の次元に射影する層
        # audio_output_dim = self.audio_encoder.config.hidden_size # Wav2Vec2
        # vision_output_dim = self.vision_encoder.config.hidden_size # ViT
        # self.audio_projection = nn.Linear(audio_output_dim, embedding_dim)
        # self.vision_projection = nn.Linear(vision_output_dim, embedding_dim)

    def forward(self, audio_inputs=None, image_inputs=None):
        audio_features = None
        if audio_inputs is not None:
            audio_features = self.audio_encoder(audio_inputs)

        image_features = None
        if image_inputs is not None:
            image_features = self.vision_encoder(image_inputs)
            
        return audio_features, image_features

# CPUで現実的に動作させるための簡易エンコーダ
class AudioLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(n_mfcc=input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, waveform):
        # (batch, time) -> (batch, n_mfcc, time)
        mfccs = self.mfcc(waveform).transpose(1, 2)
        # (batch, time, n_mfcc)
        _, (hidden, _) = self.lstm(mfccs)
        return hidden.squeeze(0)

class VisionResNetEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1] # FC層を除去
        self.resnet = nn.Sequential(*modules)
        self.projection = nn.Linear(resnet.fc.in_features, embedding_dim)

    def forward(self, images):
        features = self.resnet(images).flatten(1)
        return self.projection(features)