import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser(description="Keep only caption rows whose image exists in images-dir")
    p.add_argument("--images-dir", type=Path, default=Path("data/images"))
    p.add_argument("--captions-file", type=Path, default=Path("data/ignore_captions.txt"))
    p.add_argument("--out-file", type=Path, default=Path("data/captions.txt"))
    p.add_argument("--image-col", type=str, default="image", help="column name in captions file that contains image filename")
    p.add_argument("--keep-first-per-image", action="store_true", help="if set, keep only the first caption per image (useful when multiple captions exist)")
    args = p.parse_args()

    if not args.images_dir.exists() or not args.images_dir.is_dir():
        raise SystemExit(f"images-dir not found: {args.images_dir}")
    if not args.captions_file.exists():
        raise SystemExit(f"captions-file not found: {args.captions_file}")

    # collect image filenames and stems present in images-dir
    img_paths = [p for p in sorted(args.images_dir.iterdir()) if p.is_file()]
    img_names = set(p.name for p in img_paths)
    img_stems = set(p.stem for p in img_paths)

    df = pd.read_csv(args.captions_file)
    if args.image_col not in df.columns:
        raise SystemExit(f"'{args.image_col}' column not found in {args.captions_file}; columns: {list(df.columns)}")

    def matches_image(val):
        if pd.isna(val):
            return False
        v = str(val)
        # exact match (with extension or path)
        if Path(v).name in img_names:
            return True
        # match by stem only
        if Path(v).stem in img_stems:
            return True
        return False

    mask = df[args.image_col].apply(matches_image)
    filtered = df[mask].copy()

    if args.keep_first_per_image and args.image_col in filtered.columns:
        # normalize to stem then drop duplicates keeping first
        filtered["__stem"] = filtered[args.image_col].apply(lambda x: Path(str(x)).stem)
        filtered = filtered.drop_duplicates(subset="__stem", keep="first").drop(columns="__stem")

    filtered.to_csv(args.out_file, index=False)
    print(f"Filtered {len(filtered)} rows -> {args.out_file}")

if __name__ == "__main__":
    main()