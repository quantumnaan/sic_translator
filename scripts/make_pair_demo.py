# prepare_parallel_data.py

"""
prepare_parallel_data.py

目的:
Flickr8kのデータソースから、フェーズCのファインチューニングに
使用するための対訳データセットを構築する。

入力:
- data/captions.txt (画像ファイル名と英語キャプションのリスト)
- data/audio/ (キャプションから生成された音声ファイル群)

出力:
- data/parallel_pairs/audio/ (音声ファイルのコピー)
- data/parallel_pairs/text/ (対応する英語キャプションのテキストファイル)
"""

import os
import argparse
import pandas as pd
import shutil
from tqdm import tqdm

def get_args():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(description="Prepare parallel data for Phase C fine-tuning.")
    parser.add_argument("--captions_file", type=str, default="data/captions.txt", help="Path to the captions text file.")
    parser.add_argument("--source_audio_dir", type=str, default="data/audio/", help="Path to the directory containing source audio files.")
    parser.add_argument("--output_dir", type=str, default="data/parallel_pairs/", help="Path to the output directory for parallel data.")
    parser.add_argument("--num_samples", type=int, default=None, help="Maximum number of parallel pairs to create (for debugging). Default is all.")
    
    return parser.parse_args()

def main():
    args = get_args()

    # 1. 出力ディレクトリを作成
    output_audio_dir = os.path.join(args.output_dir, "audio")
    output_text_dir = os.path.join(args.output_dir, "text")
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_text_dir, exist_ok=True)

    print(f"Output directories created at: {args.output_dir}")

    # 2. キャプションファイルを読み込む
    try:
        df = pd.read_csv(args.captions_file)
    except FileNotFoundError:
        print(f"Error: Captions file not found at {args.captions_file}")
        return
    
    print(f"Loaded {len(df)} captions from {args.captions_file}")

    # 3. 各キャプションに対応するデータペアを作成
    created_count = 0
    # 各画像に対して複数のキャプションがあるため、重複を避けるためのセット
    processed_images = set()

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Creating parallel data"):
        if args.num_samples is not None and created_count >= args.num_samples:
            print(f"\nReached the limit of {args.num_samples} samples.")
            break

        image_filename = row['image']
        caption = row['caption']
        
        base_name = os.path.splitext(image_filename)[0]
        
        # 1つの画像につき1つのペアのみを作成する
        if base_name in processed_images:
            continue

        source_audio_path = os.path.join(args.source_audio_dir, f"{base_name}.wav")

        # 対応する音声ファイルが存在するか確認
        if os.path.exists(source_audio_path):
            # 出力先のパスを定義
            dest_audio_path = os.path.join(output_audio_dir, f"{base_name}.wav")
            dest_text_path = os.path.join(output_text_dir, f"{base_name}.txt")

            # 音声ファイルをコピー
            shutil.copy(source_audio_path, dest_audio_path)
            
            # テキストファイルを作成してキャプションを書き込む
            with open(dest_text_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            processed_images.add(base_name)
            created_count += 1

    print("\n--- Parallel Data Preparation Complete ---")
    print(f"Successfully created {created_count} audio-text pairs in '{args.output_dir}'")
    print("This directory is now ready to be used with the '--parallel_data_dir' argument in 'finetune_translator.py'.")


if __name__ == "__main__":
    main()