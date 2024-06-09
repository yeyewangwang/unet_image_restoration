"""
Load a pretrained UNet for image restoration model,
generate predictions,
save results in predicted masks, predicted images,
and reconstructed images directories.
"""
import argparse, os
import torch, tqdm
from data.dataset import create_data_loader
from data.img_io import img_batch_tensor_to_pils, mask_batch_tensor_to_pils
from model.image_restoration_model import UnetImageRestorationModel

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint',
                       type=str, required=True,
                       help="Path to model checkpoint.")
    parser.add_argument('--data-dir',
                        type=str, default="./input/original_validation",
                        help="Input data directory")
    parser.add_argument('--output-dir',
                        type=str, default='./output',
                        help='Directory to save predicted masks,'
                             'predicted images, and reconstructed images.')
    parser.add_argument('--binary_mask_threshold', type=float,
                        default=0.1, help="Threshold for converting predicted mask"
                                          "to corrupted binary mask. Predicted values"
                                          "greater or equal to this value will get rounded"
                                          "up to 1.0 and marked as a corrupted pixel.")
    args = parser.parse_args()

    model_cpt_state_dict = torch.load(args.model_checkpoint,
                                      map_location=torch.device(device))

    unet_model = UnetImageRestorationModel()
    unet_model.load_state_dict(model_cpt_state_dict)
    unet_model.eval()
    unet_model.to(device)

    predicted_imgs_dir = os.path.join(args.output_dir, "predicted_imgs")
    reconstructed_imgs_dir = os.path.join(args.output_dir, "reconstructed_imgs")
    binary_masks_dir = os.path.join(args.output_dir, "binary_masks")

    os.makedirs(predicted_imgs_dir, exist_ok=True)
    os.makedirs(reconstructed_imgs_dir, exist_ok=True)
    os.makedirs(binary_masks_dir, exist_ok=True)

    with torch.inference_mode():
        val_loader = create_data_loader(args.data_dir,
                                        batch_size=4,
                                        is_validation=True,
                                        output_img_id=True,
                                        inference_only=True)
        for data, _, _, corrupted_img_paths in tqdm.tqdm(val_loader):
            data = data.to(device)
            predicted_img, predicted_mask = unet_model(data)

            binary_mask = (predicted_mask >= args.binary_mask_threshold).int()
            expanded_binary_mask = binary_mask.repeat(1, 3, 1, 1)
            reconstructed_img = predicted_img * expanded_binary_mask + (
                        1 - expanded_binary_mask) * data

            recon_pils = img_batch_tensor_to_pils(reconstructed_img)
            mask_pils = mask_batch_tensor_to_pils(binary_mask)
            pred_pils = img_batch_tensor_to_pils(predicted_img)

            for i, corrupted_img_path in enumerate(corrupted_img_paths):
                filename = os.path.basename(corrupted_img_path)

                recon_pils[i].save(
                    os.path.join(reconstructed_imgs_dir,
                                 filename))
                pred_pils[i].save(os.path.join(predicted_imgs_dir, filename))
                mask_pils[i].save(os.path.join(binary_masks_dir, filename))
