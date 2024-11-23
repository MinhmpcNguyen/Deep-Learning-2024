### **Report**

#### **1. Transformation Techniques Used**
Transformations applied to the images before training include the following:

1. **Augmentations (Training Phase)**:
   - **HorizontalFlip (p=0.4)**
   - **VerticalFlip (p=0.4)**
   - **RandomGamma (gamma_limit=(70, 130), p=0.2)**: Adjusts the image gamma to simulate varying lighting conditions.
   - **RGBShift (p=0.3)**: Slightly shifts the RGB channels to simulate different lighting or camera effects.
   - **Normalize (mean, std)**: Scales pixel values to have a zero mean and unit variance based on the ImageNet dataset statistics.
   - **ToTensorV2()**: Converts the augmented images to PyTorch tensors.

2. **Validation Phase Transformations**:
   - **Normalize (mean, std)**: Normalizes the pixel values with the same parameters as the training set to ensure consistency.
   - **ToTensorV2()**: Converts the normalized images to PyTorch tensors.

#### **2. Model Architecture Chosen**
- **Model**: `Unet++`

- **Encoder**: `EfficientNet-B0`

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

For running on colab:

!git clone https://github.com/MinhmpcNguyen/Deep-Learning-2024.git

In this line parser.add_argument("--checkpoint", type=str, default="best_model_optimized.pth", help="Path to the model checkpoint.") replace "best_model_optimized.pth" to "/content/Deep-Learning-2024/best_model_optimized.pth"

!python3 /content/Deep-Learning-2024/infer.py --image_path image.jpeg

For running on Kaggle:

!git clone https://github.com/MinhmpcNguyen/Deep-Learning-2024.git

In this line parser.add_argument("--checkpoint", type=str, default="best_model_optimized.pth", help="Path to the model checkpoint.") replace "best_model_optimized.pth" to "/kaggle/Deep-Learning-2024/best_model_optimized.pth"

!python3 /kaggle/Deep-Learning-2024/infer.py --image_path image.jpeg

