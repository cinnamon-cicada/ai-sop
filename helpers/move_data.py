import os
import shutil
from pathlib import Path

def move(source_dir: str, target_name: str, n_per_split: int = 100, base_dir: str = "demo_dataset"):
    """
    Moves files from `source_dir` into train/val folders:
    - 100 files → {base_dir}/train/[target_name]
    - 100 files → {base_dir}/val/[target_name]
    """

    source = Path(source_dir)
    if not source.exists():
        raise FileNotFoundError(f"Source directory '{source}' does not exist")

    files = sorted([f for f in source.iterdir() if f.is_file()])
    if len(files) < n_per_split * 2:
        raise ValueError(f"Not enough files in {source_dir}. Need at least {n_per_split*2}")

    # Prepare target directories
    train_dir = Path(base_dir) / "train" / target_name
    val_dir   = Path(base_dir) / "val" / target_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Move files
    for i, f in enumerate(files[:n_per_split]):
        shutil.move(str(f), train_dir / f.name)
    for i, f in enumerate(files[n_per_split:n_per_split*2]):
        shutil.move(str(f), val_dir / f.name)

    print(f"[DONE] Moved {n_per_split} files to {train_dir}")
    print(f"[DONE] Moved {n_per_split} files to {val_dir}")


# Usage:
# move("output/construction workers laying bricks", "laying_bricks")
# move("output/construction workers installing LED lights", "installing_led_lights")
# move("output/construction workers pouring concrete", "pouring_concrete")

# # May uncomment, for more testing
# move("output/construction workers installing windows", "installing_windows")
# move("output/construction workers fixing sink plumbing", "fixing_sink_plumbing")