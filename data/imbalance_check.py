import os, argparse
import numpy as np
from typing import List


def gather_files(directory: str) -> List[str]:
    """Collect all file names (without extension) from the
    directory."""
    return [os.path.join(directory, f) for f in
            os.listdir(directory) if
            os.path.isfile(os.path.join(directory, f))]


def positive_rate(files: List[str]) -> float:
    total_ratio = 0
    num_files = 0

    ratios = []

    for file in files:
        num_files += 1

        data = np.load(file)

        if data.shape != (168, 298):
            raise ValueError(
                f"File {file} has shape {data.shape}")

        ratio = np.sum(data) / np.prod(data.shape)
        total_ratio += ratio
        ratios.append(ratio)

    print(f"Std: {np.std(ratios)}, min: {np.min(ratios)}, max: {np.max(ratios)}")
    return total_ratio / num_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src',
                        type=str,
                        required=True)
    args = parser.parse_args()
    pos_rate = positive_rate(gather_files(args.src))
    print(f"Marked pixels is on average {pos_rate}.")


