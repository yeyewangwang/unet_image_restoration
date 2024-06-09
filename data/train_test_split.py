import argparse
import os
import shutil
from typing import List
import random


def gather_files(directory: str) -> List[str]:
    """Collect all file names (without extension) from the
    directory."""
    return sorted([os.path.splitext(f)[0] for f in
            os.listdir(directory) if
            os.path.isfile(os.path.join(directory, f))], key=lambda filebase: int(filebase))


def copy_files(files: List[str], kind: str, dst_dir: str,
               paths: dict[str:str], original_validation: bool = False):
        dst_path = os.path.join(dst_dir, kind)
        os.makedirs(dst_path, exist_ok=True)
        for subdir, path in paths.items():
            os.makedirs(os.path.join(dst_path, subdir), exist_ok=True)

        for file in files:
            for subdir, path in paths.items():
                filename = file
                fileext = ('.npy' if subdir == 'binary_masks' else '.png')
                filename += fileext
                src_file = os.path.join(path, filename)
                if original_validation:
                    dst_filename = file
                    dst_filename += (
                        "_original_validation" if original_validation else "")
                    dst_filename += fileext
                    dst_file = os.path.join(dst_path,
                                            subdir,
                                            dst_filename)
                else:
                    dst_file = os.path.join(dst_path,
                                        subdir,
                                        filename)
                shutil.copy(src_file, dst_file)

                if random.random() < 5 / len(files):
                    print(
                        f"Copied file {src_file} into {dst_file}")


def create_datasets(src_dir: str, dst_dir: str, train_ratio: float,
                    test_ratio: float, seed: int = 11,
                    existing_validation: int = 25):
    """Randomly split files into training, testing, and
    validation sets."""
    # Set the random seed for reproducibility
    random.seed(seed)

    # Paths to the subdirectories
    paths = {
        'binary_masks': os.path.join(src_dir,
                                     'binary_masks'),
        'corrupted_imgs': os.path.join(src_dir,
                                       'corrupted_imgs'),
        'src_imgs': os.path.join(src_dir, 'src_imgs'),
        'motifs': os.path.join(src_dir, 'motifs')
    }

    # Ensure all directories have the same number of files
    # and they are synchronized
    files_list = [gather_files(path) for path in
                  paths.values()]
    if not all(len(files) == len(files_list[0]) for files in
               files_list):
        raise ValueError(
            "Directories do not contain the same number of "
            + "files or are not synchronized.")

    # Choose one directory to sample file names from
    all_files = list(files_list[0])

    # Calculate split indices
    total_files = len(all_files) + existing_validation
    train_end = int(total_files * train_ratio)
    test_end = train_end + int(total_files * test_ratio)

    # Split files
    train_files = all_files[:train_end]
    test_files = all_files[train_end:test_end]
    validation_files = all_files[test_end:]

    # Copy files to respective directories
    copy_files(train_files, kind='train', dst_dir=dst_dir, paths=paths)
    copy_files(test_files, kind='test', dst_dir=dst_dir, paths=paths)
    copy_files(validation_files, kind='validation', dst_dir=dst_dir, paths=paths)


def move_existing_validation(src_dir: str, dst_dir: str):
    # Paths to the subdirectories
    paths = {
        'binary_masks': os.path.join(src_dir,
                                     'binary_masks'),
        'corrupted_imgs': os.path.join(src_dir,
                                       'corrupted_imgs'),
        'src_imgs': os.path.join(src_dir, 'src_imgs'),
        'motifs': os.path.join(src_dir, 'motifs')
    }

    # Ensure all directories have the same number of files
    # and they are synchronized
    files_list = [set(gather_files(path)) for path in
                  paths.values()]
    if not all(len(files) == len(files_list[0]) for files in
               files_list):
        raise ValueError(
            "Directories do not contain the same number of " +
            "files or are not synchronized.")

    # Choose one directory to sample file names from
    all_files = list(files_list[0])
    copy_files(all_files, kind='validation', dst_dir=dst_dir,
               paths=paths, original_validation=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src',
                        type=str,
                        required=True)
    parser.add_argument('--dst',
                        type=str,
                        required=True)
    args = parser.parse_args()
    create_datasets(os.path.join(args.src, "train"),
                    args.dst,
                    train_ratio=0.8, test_ratio=0.1,
                    existing_validation=25, seed=11)
                    # train_ratio=0.76, test_ratio=0.12,
                    # existing_validation=25, seed=11)

    move_existing_validation(os.path.join(args.src, "validation"), args.dst)

    print("WARNING: please manually make sure that:\n"
          "1. the last file in training set is not the same as the first file in the testing set;\n"
          "2. the last file in testing set is not the same as the first file in the validation set;\n")
