### **README.md**

# **プロジェクト名：マルチモーダル音声翻訳システムの構築**

## **1. プロジェクト概要**

本プロジェクトは、プロトタイプ版で検証したマルチモーダル学習(readme_protoを参照)のコンセプトを拡張し、テキストデータが存在しない未知の言語（以下、ターゲット言語）の生音声から英語への翻訳を可能にする、エンドツーエンドのシステムを構築することを目的とします。

開発は以下の3つの主要フェーズで構成されます。

* **フェーズA：音響単位の発見**: 大規模な事前学習済み音声モデルを、ターゲット言語の音声データで教師なしファインチューニングし、言語固有の音響単位（疑似音素）発見器を構築します。  
* **フェーズB：意味空間のマッピング**: フェーズAで構築したモデルを使い、音声データを音響単位系列に変換後、画像データと対照学習させることで、意味の理解を可能にします。  
* **フェーズC：文法的な洗練**: 少量の高品質な対訳データを使い、モデル全体をファインチューニングして、文法的に正しく自然な翻訳を実現します。

【重要】ハードウェアに関する注意:  
本プロジェクト、特にフェーズAのファインチューニングは、大量の計算リソースを必要とします。GPU環境（NVIDIA V100, A100など）での実行を強く推奨します。 CPUベースでの実行は、ごく少量のデータでのデバッグやコード検証に限定してください。

## **2. 開発環境の構築**

**セットアップコマンド:**

```bash
# 仮想環境を作成・有効化  
python -m venv venv
source venv/bin/activate

# ライブラリのインストール  
pip install -r requirements.txt
```

## **フェーズA：音響単位の発見（教師なしファインチューニング）**

**目的**: ターゲット言語の音声に特化した、高精度な音響単位発見モデルを構築する。

### **A-1. データ準備**

* **必要なデータ**: ターゲット言語の**ラベルなし音声データ**。最低でも10時間、可能であれば100時間以上が望ましい。  
* **ディレクトリ構成**:  
  ```
  data/  
  ├── target_lang_unlabeled/  
  │   ├── audio_001.wav  
  │   ├── audio_002.wav  
  │   └── ...
  ```

まず，README_proto.mdを参照して，Flickr8kからdata/audioを作成(これはaudioとimageの埋め込みを一致させるために使用)．
今回は疑似的にこれをそのままコピーして音素発見にも使用．

### **A-2. モデル選定**

* 多言語で事前学習された大規模音声モデルを利用します。これにより、ゼロからの学習を避け、効率的にターゲット言語に適応できます。  
* **推奨モデル**: facebook/wav2vec2-base

### **A-3. 教師なしファインチューニング**

Wav2Vec2の基本アーキテクチャを、ターゲット言語の音声データを用いてさらに学習させ、その言語の音響的特徴に特化させます。

* **スクリプト**: `src/finetune_aud.py` を作成。  
* **主な機能**:  
  * transformersライブラリを使用し、facebook/wav2vec2-baseをロード。  
  * `data/target_lang_unlabeled/` から音声データを読み込み、モデルの事前学習タスク（対照損失）を継続して実行。  
  * ファインチューニング済みのベースモデルを `models/aud_base_model/` に保存。  
* **実行コマンド (CLI向け)**:  
  ```bash
  python src/finetune_aud.py \
    --base_model "facebook/wav2vec2-base" \
    --data_dir "data/target_lang_unlabeled/" \
    --output_dir "models/aud_base_model/" \
    --num_train_epochs 10 \
    --learning_rate 1e-5
  ```

### **A-4. VQレイヤーの学習**

ファインチューニング済みのベースモデルに、離散化のためのVQ（ベクトル量子化）レイヤーを追加し、学習させます。

* **スクリプト**: `src/train_vq.py` を作成。  
* **主な機能**:  
  * `models/aud_base_model/` からベースモデルをロードし、そのパラメータは凍結（更新しない）。  
  * 音声データから抽出された特徴量をK-Meansでクラスタリングし、VQレイヤーのコードブックを初期化。  
  * VQレイヤーのみを学習させ、`models/acoustic_unit_model/` に最終的なモデルを保存。  
* **実行コマンド**:  
  ```bash
  python src/train_vq.py \
    --base_model_path "models/aud_base_model/" \
    --data_dir "data/target_lang_unlabeled/" \
    --output_dir "models/acoustic_unit_model/" \
    --num_clusters 512 # コードブックサイズ (Optunaで最適化可能)
  ```

【フェーズAの成果物】: `models/acoustic_unit_model/`  
ターゲット言語の生音声を入力すると、`[5, 12, 8, ...]` という音響単位の系列を出力するモデル。

---

## **フェーズB：意味空間のマッピング（マルチモーダル学習）**

**目的**: フェーズAの成果物を使い、音響単位の系列と画像の意味を関連付ける。

### **B-1. データ準備**

* **必要なデータ**: **データ収集アプリ(App 1)**で収集した (ターゲット言語の音声, 対応する画像) のペア。  
* **前処理**:  
  * まず、このフェーズで使う全ての音声データを、フェーズAで完成した `acoustic_unit_model` に通し、**音響単位の系列に変換**しておきます。  
  * この変換結果を `data/multimodal_preprocessed/` などに保存します。

### **B-2. 対照学習の実行**

* **スクリプト**: `src/train_multimodal.py` を作成。  
* **主な機能**:  
  * **Audio Encoder**: 音響単位の系列（整数の系列）を入力として受け取るTransformerまたはLSTMモデル。  
  * **Image Encoder**: 事前学習済みのVision Transformer (ViT)など。  
  * `data/multimodal_preprocessed/` からデータを読み込み、プロトタイプ版と同様の対照学習を実行。  
* **実行コマンド**:  
  ```bash
  python src/train_multimodal.py \
    --preprocessed_data_dir "data/multimodal_preprocessed/" \
    --image_dir "data/images/" \
    --output_dir "models/semantic_core_model/" \
    --embedding_dim 256 # Optunaで最適化
  ```

【フェーズBの成果物】: `models/semantic_core_model/`  
音響単位系列を、画像と共通の意味空間ベクトルに変換できるモデル。

---

## **フェーズC：文法的な洗練（ファインチューニング）**

**目的**: フェーズBのモデルに、文法的に正しく自然な英語を出力する能力を付与する。

### **C-1. データ準備**

* **必要なデータ**: **データ収集アプリ(App 1)**で収集した (ターゲット言語の音声, 高品質な英語訳テキスト) のペア。

### **C-2. エンドツーエンドのファインチューニング**

* **スクリプト**: `src/finetune_translator.py` を作成。  
* **主な機能**:  
  * **統合モデル**: `acoustic_unit_model` + `semantic_core_model` + 英語テキストデコーダ を連結した、エンドツーエンドの翻訳モデルを構築。  
  * 対訳データを使い、モデル全体のパラメータを微調整（ファインチューニング）する。  
* **実行コマンド**:  
  ```bash
  python src/finetune_translator.py \
  --acoustic_model_path "models/acoustic_unit_model/" \
  --semantic_model_path "models/semantic_core_model/semantic_core_model.pth" \
  --params_path "models/semantic_core_model/best_params.json" \
  --parallel_data_dir "data/parallel_pairs/" \
  --output_dir "models/final_translator_model/" \
  --decoder_model_name "facebook/bart-base" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5
  ```

【フェーズCの成果物】: `models/final_translator_model/`  
ターゲット言語の生音声を入力すると、文法的に正しく自然な英語テキストを出力する最終モデル。

---

## **5. 最終的なディレクトリ構成**

```
.
├── data/
│   ├── target_lang_unlabeled/
│   ├── multimodal_pairs/   # (音声,画像)ペア
│   └── parallel_pairs/     # (音声,英語テキスト)ペア
├── models/
│   ├── acoustic_unit_model/
│   ├── semantic_core_model/
│   └── final_translator_model/
├── src/
│   ├── finetune_aud.py
│   ├── train_vq.py
│   ├── train_multimodal.py
│   └── finetune_translator.py
├── scripts/
│   └── prepare_audio_data.py # (プロトタイプ時のデータ準備スクリプトなど)
├── README.md
└── requirements.txt
```
## デモ: 音声 -> 画像 検索 (audio-to-image retrieval)

以下は `src/demo_aud2img.py` の説明です。これはフェーズA/フェーズBで学習したモデルを使い、入力音声に最も近い画像をギャラリーから検索して表示するデモプログラムです。

主な機能
- フェーズA の音響ベースモデルで波形から音響特徴を抽出し、事前に学習した k-means コードブックで離散化します。
- フェーズB（semantic core）で音響単位系列を意味ベクトルに変換します。
- 画像ギャラリーの各画像を画像エンコーダでベクトル化してインデックス化し、クエリ音声とのコサイン類似度で近い画像を返します。

使い方（例）
```bash
python src/demo_aud2img.py \
  --acoustic_model_path models/acoustic_unit_model/ \
  --model_dir models/semantic_core_model/ \
  --image_gallery_dir data/images/ \
  --query_audio_file data/audio/10815824_2997e03d76.wav \
  --top_k 5
```

主要な引数
- `--acoustic_model_path`: フェーズA の出力ディレクトリ（kmeans コードブック等が存在する）
- `--model_dir` / `--semantic_model_path`: フェーズB の学習済みモデルディレクトリ／重みパス（`semantic_core_model.pth` や `best_params.json` が必要）
- `--image_gallery_dir`: 検索対象の画像フォルダ
- `--query_audio_file`: 検索に使うクエリ音声ファイル（wav）
- `--top_k`: 返す上位 k 件

出力と挙動
- 実行すると、まず画像ギャラリーのベクトル化結果を `image_index.pt`（semantic モデルの場所）に保存またはロードします。
- クエリ音声はフェーズA→kmeans→フェーズB の順で意味ベクトル化され、画像ベクトルとコサイン類似度で比較されます。
- 上位 k 件の画像と類似度スコアを標準出力に表示し、画像を matplotlib で並べて表示します。

依存関係 / 前提
- PyTorch / torchaudio / torchvision
- 学習済みのフェーズA（`models/acoustic_unit_model/`）とフェーズB（`models/semantic_core_model/semantic_core_model.pth`）
- `joblib`（kmeans のロード用）
- 画像前処理関数（リポジトリ内の `dataloader.get_transforms()` など）

注意事項
- モデルのロードや画像インデックス作成は GPU が使えると高速ですが、CPU でも動作します（ただし時間がかかる）。
- `image_index.pt` は semantic モデルのディレクトリに保存されます。モデル構成を変えた場合は再作成してください。
- 音声・画像モデルはリポジトリに大きなバイナリを置かない運用（Hugging Face / S3）を推奨します。

トラブルシュート
- モデル読み込みで Unauthorized / 401 が出る場合は Hugging Face トークンを環境変数 `HUGGINGFACE_HUB_TOKEN` にセットするか、ローカルにモデルを配置してください。
- メモリ不足で落ちる場合は小さいモデルを使う、あるいはバッチサイズ／worker 数を下げて実行してください。

## phase C 後のデモ
```bash
python src/demo_translator.py \
  --model_path "models/final_translator_model/" \
  --audio_file "data/audio/10815824_2997e03d76.wav"
```