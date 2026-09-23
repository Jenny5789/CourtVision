import os
import glob

DATASET_DIR = "data/ball_dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")

VAL_VIDEOS = ["video9", "video10"]

all_images = glob.glob(os.path.join(IMG_DIR, "*.jpg"))

train_list = []
val_list = []

for img_path in all_images:
    filename = os.path.basename(img_path)
    abs_path = os.path.abspath(img_path)
    if any(filename.startswith(v + "_") for v in VAL_VIDEOS):
        val_list.append(abs_path)
    else:
        train_list.append(abs_path)

with open(os.path.join(DATASET_DIR, "train.txt"), "w") as f:
    f.write("\n".join(train_list))

with open(os.path.join(DATASET_DIR, "val.txt"), "w") as f:
    f.write("\n".join(val_list))

print(f"Train: {len(train_list)}장")
print(f"Val: {len(val_list)}장")