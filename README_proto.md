# **プロジェクト名：マルチモーダル学習による音声-画像意味空間の構築**

## **1\. プロジェクト概要**

本プロジェクトの目的は、音声データと画像データを意味的に関連付けるコアモデルを構築することです。具体的には、ある音声とその音声が描写する内容の画像を、共通のベクトル空間（意味空間）内の近い位置にマッピングするモデルを学習させます。

このコアモデルは、将来的により高度な翻訳システム（例：未知の言語の音声から英語のテキスト/画像を生成）の基盤となります。本ドキュメントでは、プロトタイピングとして**英語をソース言語**とみなし、CPUベースでの開発手順を詳述します。

**コア技術**:

* **マルチモーダル学習**: 音声と画像、異なるモダリティのデータを扱います。  
* **対照学習 (Contrastive Learning)**: 正しいペア（音声とその内容を表す画像）のベクトルを近づけ、間違ったペアのベクトルを遠ざけることで、意味的な類似性を学習します。  
* **ハイパーパラメータ最適化**: Optuna を使用し、学習率などの主要なパラメータを自動で最適化します。

## **2\. 開発環境の構築**

#### **2.1. 前提条件**

* Python 3.9 以上  
* pip (Pythonパッケージインストーラ)  
* ffmpeg (音声ファイルの処理に必要。OSに応じてインストールしてください)

#### **2.2. 必要なライブラリ**

以下の内容で requirements.txt ファイルを作成してください。

Plaintext

\# torch-cpu: CPUベースでの開発のため、torchはCPU版を明示的に指定します  
torch  
torchvision  
torchaudio

\# ML / DL Framework & Tools  
transformers  
datasets  
optuna  
scikit-learn

\# Data Handling & Utilities  
pandas  
numpy  
Pillow  
librosa  
tqdm

\# Text-to-Speech for data simulation  
gTTS

#### **2.3. 環境設定コマンド**

ターミナルで以下のコマンドを実行し、環境をセットアップします。

Bash

\# 仮想環境の作成と有効化 (推奨)  
python \-m venv venv  
source venv/bin/activate  \# macOS/Linux  
\# venv\\Scripts\\activate    \# Windows

\# 必要なライブラリのインストール  
pip install \-r requirements.txt

## **3\. データセットの準備**

標準的なマルチモーダルデータセットである **Flickr8k** を使用します。このデータセットには画像と、その画像を説明する英語キャプションが含まれています。今回は「音声データ」をシミュレートするために、このキャプションからTTS（Text-to-Speech）で音声ファイルを生成します。

#### **ステップ1：Flickr8kデータセットのダウンロード**

1. [こちらのKaggleページ](https://www.kaggle.com/datasets/adityajn105/flickr8k)などからデータセットをダウンロードし、解凍します。  
2. プロジェクトルートに data/ ディレクトリを作成し、その中に images/ と captions.txt を配置します。

#### **ステップ2：キャプションから音声データを生成**

captions.txt を読み込み、各キャプションに対応する音声ファイル（.wav）を生成します。以下の内容で prepare\_audio\_data.py を作成し、実行してください。

Python

\# prepare\_audio\_data.py  
import pandas as pd  
from gtts import gTTS  
from tqdm import tqdm  
import os

\# 出力ディレクトリの作成  
audio\_dir \= "data/audio"  
os.makedirs(audio\_dir, exist\_ok=True)

\# キャプションデータの読み込み  
df \= pd.read\_csv("data/captions.txt")

\# 各キャプションから音声を生成  
print("Generating audio files from captions...")  
for index, row in tqdm(df.iterrows(), total=df.shape\[0\]):  
    image\_id \= row\['image'\]  
    caption \= row\['caption'\]  
      
    \# ファイル名を画像IDとキャプションインデックスで一意にする  
    \# 例: 1000268201\_693b08cb0e.jpg の最初のキャプション \-\> 1000268201\_693b08cb0e\_0.wav  
    base\_image\_name \= os.path.splitext(image\_id)\[0\]  
      
    \# 同じ画像に複数のキャプションがあるため、重複を避ける  
    \# 簡単のため、各画像の最初のキャプションのみを使用する  
    \# （本格的に行う場合は全キャプションを処理する）  
    if not os.path.exists(os.path.join(audio\_dir, f"{base\_image\_name}.wav")):  
        try:  
            tts \= gTTS(text=caption, lang='en')  
            tts.save(os.path.join(audio\_dir, f"{base\_image\_name}.wav"))  
        except Exception as e:  
            print(f"Skipping {base\_image\_name} due to an error: {e}")

print("Audio data generation complete.")

**実行コマンド:**

Bash

python prepare\_audio\_data.py

#### **最終的なディレクトリ構成**

.  
├── data/  
│   ├── images/  
│   │   ├── 1000268201\_693b08cb0e.jpg  
│   │   └── ...  
│   ├── audio/  
│   │   ├── 1000268201\_693b08cb0e.wav  
│   │   └── ...  
│   └── captions.txt  
├── prepare\_audio\_data.py  
└── requirements.txt  
... (今後作成するファイル)

## **4\. 開発ワークフロー**

開発をモジュール化して進めます。

### **4.1. model.py：モデルアーキテクチャの定義**

音声エンコーダと画像エンコーダを一つのクラスにまとめます。CPUでの実行可能性を考慮し、比較的小さな事前学習済みモデルを基盤とします。

Python

\# model.py  
import torch  
import torch.nn as nn  
from transformers import Wav2Vec2Model, ViTModel  
import torchaudio

class MultimodalModel(nn.Module):  
    def \_\_init\_\_(self, audio\_model\_name, vision\_model\_name, embedding\_dim):  
        super(MultimodalModel, self).\_\_init\_\_()  
        \# CPUでの実行を考慮し、モデルサイズに注意  
        \#self.audio\_encoder \= Wav2Vec2Model.from\_pretrained(audio\_model\_name)  
        \#self.vision\_encoder \= ViTModel.from\_pretrained(vision\_model\_name)  
          
        \# CPU向け簡易版: MFCC \+ LSTM  
        self.audio\_encoder \= AudioLSTMEncoder(input\_size=40, hidden\_size=embedding\_dim)  
        \# CPU向け簡易版: 事前学習済みResNet  
        self.vision\_encoder \= VisionResNetEncoder(embedding\_dim=embedding\_dim)

        \# 最終的な特徴量を共通の次元に射影する層  
        \# audio\_output\_dim \= self.audio\_encoder.config.hidden\_size \# Wav2Vec2  
        \# vision\_output\_dim \= self.vision\_encoder.config.hidden\_size \# ViT  
        \# self.audio\_projection \= nn.Linear(audio\_output\_dim, embedding\_dim)  
        \# self.vision\_projection \= nn.Linear(vision\_output\_dim, embedding\_dim)

    def forward(self, audio\_inputs, image\_inputs):  
        \# audio\_features \= self.audio\_encoder(audio\_inputs).last\_hidden\_state.mean(dim=1)  
        \# image\_features \= self.vision\_encoder(image\_inputs).last\_pooler\_output  
          
        audio\_features \= self.audio\_encoder(audio\_inputs)  
        image\_features \= self.vision\_encoder(image\_inputs)  
          
        \# audio\_embedding \= self.audio\_projection(audio\_features)  
        \# image\_embedding \= self.vision\_projection(image\_features)  
          
        return audio\_features, image\_features

\# CPUで現実的に動作させるための簡易エンコーダ  
class AudioLSTMEncoder(nn.Module):  
    def \_\_init\_\_(self, input\_size, hidden\_size):  
        super().\_\_init\_\_()  
        self.mfcc \= torchaudio.transforms.MFCC(n\_mfcc=input\_size)  
        self.lstm \= nn.LSTM(input\_size, hidden\_size, batch\_first=True)  
      
    def forward(self, waveform):  
        \# (batch, time) \-\> (batch, n\_mfcc, time)  
        mfccs \= self.mfcc(waveform).transpose(1, 2)  
        \# (batch, time, n\_mfcc)  
        \_, (hidden, \_) \= self.lstm(mfccs)  
        return hidden.squeeze(0)

class VisionResNetEncoder(nn.Module):  
    def \_\_init\_\_(self, embedding\_dim):  
        super().\_\_init\_\_()  
        resnet \= torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  
        modules \= list(resnet.children())\[:-1\] \# FC層を除去  
        self.resnet \= nn.Sequential(\*modules)  
        self.projection \= nn.Linear(resnet.fc.in\_features, embedding\_dim)

    def forward(self, images):  
        features \= self.resnet(images).flatten(1)  
        return self.projection(features)

### **4.2. dataloader.py：データローダーの作成**

(音声, 画像) のペアをバッチ単位で読み込み、モデルに入力できる形式に前処理するクラスを定義します。

### **4.3. train.py：Optunaを用いた学習スクリプト**

学習ループとOptunaによる最適化を実装します。

Python

\# train.py  
import optuna  
import torch  
import torch.nn.functional as F  
from torch.utils.data import DataLoader  
\# dataloader.py, model.py から自作クラスをインポート

\# \--- CPUに関する警告 \---  
\# 以下の学習プロセスはCPUでは非常に長い時間がかかります。  
\# まずはデータセットのサブセット（例: 100件）でコードが  
\# 正常に動作することを確認してから、本番の学習を行ってください。  
\# \--------------------

def calculate\_contrastive\_loss(audio\_emb, image\_emb, temperature=0.07):  
    \# InfoNCE損失（CLIPで使われる損失関数）を計算  
    logits \= (audio\_emb @ image\_emb.T) / temperature  
    labels \= torch.arange(len(audio\_emb))  
    loss\_a \= F.cross\_entropy(logits, labels)  
    loss\_i \= F.cross\_entropy(logits.T, labels)  
    return (loss\_a \+ loss\_i) / 2

def objective(trial):  
    \# 1\. ハイパーパラメータの定義  
    lr \= trial.suggest\_float("lr", 1e-5, 1e-3, log=True)  
    embedding\_dim \= trial.suggest\_categorical("embedding\_dim", \[128, 256, 512\])  
    batch\_size \= trial.suggest\_categorical("batch\_size", \[16, 32\])  
      
    \# 2\. モデル、データローダー、オプティマイザのセットアップ  
    \# (dataloader.py, model.py を使ってインスタンス化)  
    \# ...  
      
    \# 3\. 学習ループ  
    \# for epoch in range(NUM\_EPOCHS):  
    \#     for batch in train\_loader:  
    \#         \# ... training step ...  
    \#         loss.backward()  
    \#         optimizer.step()

    \# 4\. バリデーションと結果の報告  
    \# val\_loss \= ...  
    \# trial.report(val\_loss, epoch)  
    \# if trial.should\_prune():  
    \#     raise optuna.exceptions.TrialPruned()

    \# return val\_loss  
    return 0.1 \# ダミーの返り値

if \_\_name\_\_ \== "\_\_main\_\_":  
    study \= optuna.create\_study(direction="minimize")  
    study.optimize(objective, n\_trials=50) \# 50回の試行で最適化  
      
    print("Best trial:")  
    trial \= study.best\_trial  
    print(f"  Value: {trial.value}")  
    print("  Params: ")  
    for key, value in trial.params.items():  
        print(f"    {key}: {value}")

**実行コマンド:**

Bash

\# 学習とハイパーパラメータ最適化を開始  
python train.py

## **5\. 次のステップ：翻訳への応用**

このコアモデルが完成した後、以下のようなステップで翻訳機能へと拡張できます。

1. 学習済みモデルを使い、データセット内の全画像と全音声を意味ベクトルに変換し、保存します。  
2. 新しい音声が入力されたら、その音声の意味ベクトルを計算します。  
3. ベクトル空間内で、入力音声のベクトルに最も近い**画像ベクトル**を探索します（最近傍探索）。  
4. その画像に対応する**英語キャプション**を翻訳結果として提示します。これは「類似例に基づく翻訳」と言えます。

より高度な翻訳（例：音声→音声）には、今回構築した意味空間を利用して、別途デコーダモデルや音声合成モデルを学習させる必要があります。

---

