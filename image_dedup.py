from imagededup.methods import CNN, PHash
import argparse
import os
import shutil

argparser = argparse.ArgumentParser()
argparser.add_argument("--path",'-p' ,type=str, default="images", help="path to image folder")

phasher = PHash()
cnn_encoder = CNN()
args = argparser.parse_args()
path = args.path

encodings = phasher.encode_images(image_dir=path)
# encodings = cnn_encoder.encode_images(image_dir=path)
duplicates = phasher.find_duplicates(encoding_map=encodings)
# duplicates = cnn_encoder.find_duplicates(image_dir=path)

def move_duplicates(duplicates, source_dir, target_dir):
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for original, dup_list in duplicates.items():
        for dup in dup_list:
            source_path = os.path.join(source_dir, dup)
            target_path = os.path.join(target_dir, dup)
            # 移动文件
            if os.path.exists(source_path):
                shutil.move(source_path, target_path)
                print(f"Moved {dup} to {target_dir}")

source_dir = path
target_dir = os.path.join(path, "duplicates")

move_duplicates(duplicates, source_dir, target_dir)

