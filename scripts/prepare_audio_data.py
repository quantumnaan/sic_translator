# prepare_audio_data.py
import pandas as pd
from gtts import gTTS
from tqdm import tqdm
import os

# 出力ディレクトリの作成
audio_dir = "data/audio"
os.makedirs(audio_dir, exist_ok=True)

# キャプションデータの読み込み
df = pd.read_csv("data/captions.txt")

# 各キャプションから音声を生成
print("Generating audio files from captions...")
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    image_id = row['image']
    caption = row['caption']
    
    # ファイル名を画像IDとキャプションインデックスで一意にする
    # 例: 1000268201_693b08cb0e.jpg の最初のキャプション -> 1000268201_693b08cb0e_0.wav
    base_image_name = os.path.splitext(image_id)[0]
    
    # 同じ画像に複数のキャプションがあるため、重複を避ける
    # 簡単のため、各画像の最初のキャプションのみを使用する
    # （本格的に行う場合は全キャプションを処理する）
    if not os.path.exists(os.path.join(audio_dir, f"{base_image_name}.wav")):
        try:
            tts = gTTS(text=caption, lang='en')
            tts.save(os.path.join(audio_dir, f"{base_image_name}.wav"))
        except Exception as e:
            print(f"Skipping {base_image_name} due to an error: {e}")

print("Audio data generation complete.")