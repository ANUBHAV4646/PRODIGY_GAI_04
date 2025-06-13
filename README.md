# PRODIGY_GAI_04

## Neural Style Transfer with PyTorch  
**Task 4 - Prodigy GAI Internship**

This project demonstrates how to perform Neural Style Transfer using PyTorch. Neural Style Transfer is a fascinating application of deep learning where the style of one image is transferred onto the content of another, creating a stylized output image.

✅ This project is part of the **Prodigy GAI Internship – Task 4**.

---

## Objective

The objective of this task is to implement a neural style transfer pipeline that blends the content of one image (e.g., a landscape) with the style of another image (e.g., a famous painting) to generate a new artistic image.

---

## Key Concepts Used

### Content vs Style
- **Content Image**: The base image whose structure we want to preserve.  
- **Style Image**: The artistic image whose style (colors, textures) we want to extract and apply.

### Pretrained VGG19 Network
- A VGG19 convolutional neural network pretrained on ImageNet is used.
- Certain intermediate layers are chosen to extract features:
  - Early layers capture style (colors, brush strokes).
  - Deeper layers retain content (shapes, structure).

### Loss Functions
- **Content Loss**: Measures how similar the generated image is to the content image.
- **Style Loss**: Measures the similarity between the style features (via Gram Matrix) of generated and style images.
- **Total Loss** = `content_weight * content_loss + style_weight * style_loss`

### Optimization
- The generated image is initialized as a copy of the content image.
- Using gradient descent (Adam Optimizer), we iteratively update the generated image to minimize the total loss.

---

## Step-by-Step Explanation

### 1. Library Imports
Used essential libraries:
- `torch`, `torchvision`
- `PIL` for image processing
- `matplotlib` for visualization
- `os`, `datetime` for file management

### 2. Load and Preprocess Images
- Define transformations to resize and normalize images to match VGG19 input.
- Load images using PIL, convert them to tensors, and move them to the appropriate device (CPU or GPU).

### 3. Feature Extraction from VGG19
- Load the VGG19 model from `torchvision.models`.
- Extract specific layers to compute content and style features:
  - Style: `conv1_1`, `conv2_1`, ..., `conv5_1`
  - Content: `conv4_2`

### 4. Gram Matrix Calculation
- A Gram matrix is used to capture style by measuring correlations between feature maps.
- Computed as:  
  `G = F * Fᵀ`

### 5. Loss Functions
- Defined custom functions:
  - `calculate_content_loss()`
  - `calculate_style_loss()` using Gram matrices
- Content and style weights are tunable parameters (e.g., `1e4` for content, `1e2` for style).

### 6. Style Transfer Loop
- The generated image is optimized over `num_steps` (e.g., 300 iterations).
- At each step:
  - Extract features
  - Compute content and style loss
  - Combine them to get total loss
  - Apply gradients via the optimizer
- Optionally display intermediate results

### 7. Output & Save Image
- The final stylized image is displayed using `matplotlib`.
- Automatically saved locally as a `.png` file with a timestamped filename.
- If using Google Colab, the image can be downloaded using `google.colab.files`.

---

## How to Run

1. Open the notebook in **Google Colab** or any Jupyter environment.
2. Upload your **content** and **style** images (JPG or PNG).
3. Run the cells in sequence.
4. The output image will be generated, saved, and optionally downloaded.

---

## Requirements

If running locally, install the dependencies using:

```bash
pip install torch torchvision matplotlib pillow
