import os
import shutil
import random

def split_data(img_dir, label_dir, out_dir, train_rate=0.8, val_rate=0.1, test_rate=0.1):
    images = os.listdir(img_dir)
    labels = os.listdir(label_dir)
    images_no_ext = {os.path.splitext(img)[0]: img for img in images}
    labels_no_ext = {os.path.splitext(lbl)[0]: lbl for lbl in labels}
    matched = [(name, images_no_ext[name], labels_no_ext[name]) for name in images_no_ext if name in labels_no_ext]

    # Print unmatched files
    unmatched_images = [images_no_ext[name] for name in images_no_ext if name not in labels_no_ext]
    unmatched_labels = [labels_no_ext[name] for name in labels_no_ext if name not in images_no_ext]
    if unmatched_images:
        print("Unmatched image files:")
        for img in unmatched_images:
            print(img)
    if unmatched_labels:
        print("Unmatched label files:")
        for lbl in unmatched_labels:
            print(lbl)

    random.shuffle(matched)
    total = len(matched)
    train_data = matched[:int(train_rate * total)]
    val_data = matched[int(train_rate * total):int((train_rate + val_rate) * total)]
    test_data = matched[int((train_rate + val_rate) * total):]

    # Create output folders
    for subset in ['train', 'val', 'test']:
        img_out = os.path.join(out_dir, subset, 'images')
        label_out = os.path.join(out_dir, subset, 'labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)

    # Copy files to split folders
    def copy_files(data, subset):
        img_out = os.path.join(out_dir, subset, 'images')
        label_out = os.path.join(out_dir, subset, 'labels')
        for _, img_file, label_file in data:
            shutil.copy(os.path.join(img_dir, img_file), os.path.join(img_out, img_file))
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(label_out, label_file))

    copy_files(train_data, 'train')
    copy_files(val_data, 'val')
    copy_files(test_data, 'test')
    print("Dataset splitting completed.")

if __name__ == '__main__':
    img_dir = os.path.expanduser('datasets/images')       # image folder
    label_dir = os.path.expanduser('datasets/labels_txt') # label folder
    out_dir = os.path.expanduser('datasets')              # output folder
    split_data(img_dir, label_dir, out_dir, train_rate=0.8, val_rate=0.1, test_rate=0.1)