import argparse
from pathlib import Path
import shutil

def main():
    p = argparse.ArgumentParser(description="Keep first N images (by filename sort); move the rest to an archive dir.")
    p.add_argument("--images-dir", required=True, type=Path, help="images directory")
    p.add_argument("--keep", type=int, default=1000, help="number of files to keep (first N after sorting)")
    p.add_argument("--archive-dir", type=Path, default=None, help="where to move removed files (default: <images-dir>/_removed_<timestamp>)")
    p.add_argument("--dry-run", action="store_true", help="show actions but do not move files")
    args = p.parse_args()

    images_dir = args.images_dir
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"images-dir not found: {images_dir}")

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(sorted(images_dir.glob(e)))
    files = sorted(set(files), key=lambda p: p.name)  # sort by filename, unique

    keep_n = max(0, args.keep)
    to_keep = files[:keep_n]
    to_move = files[keep_n:]

    if args.archive_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = images_dir.parent / f"{images_dir.name}_removed_{ts}"
    else:
        archive_dir = args.archive_dir

    print(f"Images dir: {images_dir}")
    print(f"Total images matched: {len(files)}")
    print(f"Keeping first {len(to_keep)} files; moving {len(to_move)} to: {archive_dir}")
    if args.dry_run:
        for pth in to_move[:20]:
            print("MOVE (dry):", pth.name)
        if len(to_move) > 20:
            print("... (showing first 20 of to_move)")
        return

    if to_move:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for pth in to_move:
            dest = archive_dir / pth.name
            # if destination exists, append numeric suffix
            if dest.exists():
                base = dest.stem
                suffix = dest.suffix
                i = 1
                while True:
                    new = archive_dir / f"{base}_{i}{suffix}"
                    if not new.exists():
                        dest = new
                        break
                    i += 1
            shutil.move(str(pth), str(dest))
    print("Done.")

if __name__ == "__main__":
    main()