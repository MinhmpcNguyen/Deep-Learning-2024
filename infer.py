import os
import numpy as np
import cv2
import torch
from albumentations import Normalize
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description="Segment an input image using the trained model.")
parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
parser.add_argument("--output_path", type=str, default="segmented_output.jpeg", help="Path to save the segmented image.")
parser.add_argument("--checkpoint", type=str, default="best_model_optimized.pth", help="Path to the model checkpoint.")
args = parser.parse_args()

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLOR_DICT = {
    0: (0, 0, 0),       # Background
    1: (255, 0, 0),     # Class 1
    2: (0, 255, 0)      # Class 2
}

# Function to convert a mask to RGB
def mask_to_rgb(mask, color_dict):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_dict.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask

# Load Model
def load_model(checkpoint_path):
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",  # Ensure consistency with training
        encoder_weights=None,
        in_channels=3,
        classes=3
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

# Perform Inference
def segment_image(image_path, model, color_dict, output_path):
    # Load and preprocess the image
    ori_img = cv2.imread(image_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = ori_img.shape[:2]

    # Resize and normalize the image
    resized_img = cv2.resize(ori_img, (256, 256))
    transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transformed = transform(image=resized_img)["image"]
    tensor_img = ToTensorV2()(image=transformed)["image"].unsqueeze(0).to(DEVICE)

    # Model inference
    with torch.no_grad():
        output = model(tensor_img)
        output_mask = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Post-process the output mask
    output_mask = cv2.resize(output_mask, (ori_w, ori_h))
    final_mask = np.argmax(output_mask, axis=2)
    segmented_image = mask_to_rgb(final_mask, color_dict)

    # Save the output
    segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, segmented_image_bgr)
    print(f"Segmented image saved to: {output_path}")

# Main Execution
if __name__ == "__main__":
    if not os.path.exists(args.image_path):
        print(f"Error: The file '{args.image_path}' does not exist.")
        exit(1)

    model = load_model(args.checkpoint)
    segment_image(args.image_path, model, COLOR_DICT, args.output_path)
