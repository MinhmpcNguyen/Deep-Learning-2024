### **Report**

#### **1. Transformation Techniques Used**
Transformations applied to the images before training include the following:

1. **Augmentations (Training Phase)**:
   - **HorizontalFlip (p=0.4)**: Randomly flips the image horizontally to add spatial invariance.
   - **VerticalFlip (p=0.4)**: Randomly flips the image vertically for further data augmentation.
   - **RandomGamma (gamma_limit=(70, 130), p=0.2)**: Adjusts the image gamma to simulate varying lighting conditions.
   - **RGBShift (p=0.3)**: Slightly shifts the RGB channels to simulate different lighting or camera effects.
   - **Normalize (mean, std)**: Scales pixel values to have a zero mean and unit variance based on the ImageNet dataset statistics.
   - **ToTensorV2()**: Converts the augmented images to PyTorch tensors.

2. **Validation Phase Transformations**:
   - **Normalize (mean, std)**: Normalizes the pixel values with the same parameters as the training set to ensure consistency.
   - **ToTensorV2()**: Converts the normalized images to PyTorch tensors.

#### **2. Model Architecture Chosen**
- **Model**: `Unet++`
  - **Reason for Choice**:
    - Unet++ is a robust and advanced segmentation architecture built upon the U-Net model.
    - It introduces dense skip connections and redesigned skip pathways to improve feature reuse and semantic understanding.
    - Suitable for precise segmentation tasks like this, especially in medical imaging or multi-class segmentation.

- **Encoder**: `EfficientNet-B0`
  - Pretrained on ImageNet, this encoder is highly efficient and computationally lightweight while providing strong feature extraction capabilities.
  - Chosen to leverage the transfer learning benefits for robust feature representation.

- **Input Channels**: `3` (for RGB images).
- **Output Classes**: `3`
  - Class `0`: Background.
  - Class `1`: Red Mask (e.g., denoting certain regions of interest).
  - Class `2`: Green Mask (another region of interest).

#### **3. Additional Techniques**
- **Resize**:
  - Input images and masks were resized to `(256, 256)` for training and inference to reduce computational cost and standardize input dimensions.
  
- **Mixed Precision Training**:
  - Used `torch.cuda.amp` with `GradScaler` for faster training and reduced memory usage.

- **Loss Function**:
  - `CrossEntropyLoss`: Chosen for multi-class segmentation, this loss function computes pixel-wise class predictions against true labels.

- **Optimization**:
  - `Adam`: An adaptive learning rate optimizer known for its fast convergence and efficiency.

git clone https://github.com/MinhmpcNguyen/Deep-Learning-2024.git

cd Deep-Learning-2024

python3 infer.py --image_path image.jpeg

