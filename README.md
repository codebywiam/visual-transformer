# ViT for Bean Leaf Disease Classification 🌱

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codebywiam/data-science-projects/blob/main/visual-transformer/vit-beans-classifier.ipynb)

A deep learning project using Vision Transformer (ViT) to classify bean leaf diseases. This implementation fine-tunes a pre-trained ViT model on the Beans Dataset from Hugging Face to identify healthy bean leaves and detect various disease conditions.


## Overview

This project implements image classification for bean leaf disease detection using a Vision Transformer (ViT) model. The model can classify bean leaf images into three categories:
- **Healthy** leaves
- **Angular Leaf Spot** disease
- **Bean Rust** disease

The implementation achieves excellent performance with **95.3% accuracy** on the test set and **100% accuracy** on the validation set during training.

## Dataset

The project uses the **Beans dataset** from Hugging Face, which contains:
- High-quality images of bean leaves
- Three balanced classes representing different health conditions
- Pre-split training, validation, and test sets
- Diverse lighting conditions and leaf orientations

### Dataset Statistics
- **Training Set**: ~1,034 images
- **Validation Set**: ~133 images  
- **Test Set**: ~128 images
- **Classes**: 3 (Angular Leaf Spot, Bean Rust, Healthy)

## Model Architecture

- **Base Model**: `google/vit-base-patch16-224`
- **Architecture**: Vision Transformer with 16x16 patches
- **Input Size**: 224x224 pixels
- **Patch Size**: 16x16
- **Fine-tuning**: Classification head adapted for 3 classes

## Dependencies

- `datasets`: Hugging Face datasets library
- `transformers`: Hugging Face transformers library
- `torch`: PyTorch deep learning framework
- `torchvision`: Computer vision utilities
- `matplotlib`: Plotting and visualization
- `scikit-learn`: Machine learning utilities
- `evaluate`: Model evaluation metrics
- `Pillow`: Image processing

## Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install datasets transformers evaluate torchvision matplotlib scikit-learn
```

## Usage

### 1. Clone and Setup

```bash
git clone <repository-url>
cd vit-bean-disease-classification
```

### 2. Run Training

**Option 1: Google Colab (Recommended)**
Click the "Open in Colab" badge at the top of this README to run the notebook directly in your browser with free GPU access.

**Option 2: Local Jupyter**
Open the Jupyter notebook `vit-beans-classifier.ipynb` and run all cells:

```bash
jupyter notebook vit-beans-classifier.ipynb
```

### 3. Make Predictions

```python
# Load trained model
model = ViTForImageClassification.from_pretrained("vit-beans-model")
image_processor = AutoImageProcessor.from_pretrained("vit-beans-model")

# Classify new images
predictions = model(processed_image)
```

## Results

### Model Performance
- **Test Accuracy**: 95.31%
- **Validation Accuracy**: 100% (final epoch)
- **Training Loss**: 0.095 (final)
- **Validation Loss**: 0.011 (final)

### Training Progress
| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|-----------------|----------|
| 1     | 0.219         | 0.086          | 96.99%   |
| 2     | 0.042         | 0.026          | 99.25%   |
| 3     | 0.005         | 0.016          | 99.25%   |
| 4     | 0.004         | 0.010          | 100.00%  |
| 5     | 0.002         | 0.011          | 100.00%  |

### Confusion Matrix Analysis

The confusion matrix reveals exceptional classification performance:

| True Label | Angular Leaf Spot | Bean Rust | Healthy | Total |
|------------|-------------------|-----------|---------|-------|
| **Angular Leaf Spot** | 41 | 2 | 0 | 43 |
| **Bean Rust** | 2 | 41 | 0 | 43 |
| **Healthy** | 0 | 2 | 40 | 42 |

**Per-Class Performance:**
- **Angular Leaf Spot**: 95.3% precision (41/43 correct)
- **Bean Rust**: 95.3% precision (41/43 correct) 
- **Healthy**: 95.2% precision (40/42 correct)

**Key Observations:**
- No confusion between diseased and healthy leaves
- Minimal misclassification only occurs between the two disease types
- Perfect separation of healthy vs. diseased categories
- Balanced performance across all three classes

## 📁 Project Structure

```
visual-transformer/
├── vit-beans-classifier.ipynb     # Main training notebook
├── README.md                     # This file
├── LICENSE

```

## Key Features

- **State-of-the-art Architecture**: Uses Vision Transformer, a cutting-edge model for image classification
- **Transfer Learning**: Leverages pre-trained weights for faster convergence
- **Comprehensive Evaluation**: Includes accuracy metrics, confusion matrix, and visual predictions
- **Data Visualization**: Provides class distribution analysis and sample image displays
- **Easy Deployment**: Saved model can be easily loaded for inference
- **Robust Preprocessing**: Automated image processing pipeline using Transformers library

## Configuration

### Training Parameters
- **Batch Size**: 16 (train/eval)
- **Learning Rate**: 2e-5
- **Epochs**: 5
- **Optimizer**: AdamW (default)
- **Scheduler**: Linear decay

### Model Specifications
- **Input Resolution**: 224×224
- **Patch Size**: 16×16
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Layers**: 12

## Dependencies

- `datasets`: Hugging Face datasets library
- `transformers`: Hugging Face transformers library
- `torch`: PyTorch deep learning framework
- `torchvision`: Computer vision utilities
- `matplotlib`: Plotting and visualization
- `scikit-learn`: Machine learning utilities
- `evaluate`: Model evaluation metrics
- `Pillow`: Image processing

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Hugging Face for the Beans dataset and Transformers library
- Google Research for the Vision Transformer architecture
- The open-source community for the various tools and libraries used

---