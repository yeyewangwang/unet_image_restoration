import os, argparse
import torch


def load_checkpoint(path: str):
    # Extract the last directory from the path
    checkpoint_dir = os.path.basename(os.path.dirname(path))

    checkpoint = torch.load(path)
    epoch, batch_idx = checkpoint['epoch'], checkpoint['batch_idx']

    new_path = os.path.join(checkpoint_dir,
                 f'inference_checkpoint_{epoch}_{batch_idx}.pt')
    torch.save(checkpoint['model_state_dict'], new_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a checkpoint and save the model state dictionary.")
    parser.add_argument('--path', type=str, required=True,
                        help="Path to the checkpoint file.")

    args = parser.parse_args()
    load_checkpoint(args.path, args.checkpoint_dir)
