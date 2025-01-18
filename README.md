# MRI-Image-Recovery
This project solves an imputation problem where we created a neural network that learns how to recover missing portions of an image

# 🧠 MRI Image Recovery: Deep Learning Assessment

## Project Overview
This repository contains my solution to an assessment focused on designing a neural network for recovering missing portions of MRI images. The primary goal is to train an architecture that can reconstruct corrupted brain images, a problem relevant in medical imaging where scans are often incomplete to save time or reduce patient exposure.

---

## 📄 Problem Statement
The dataset provided includes:
- **Corrupted Images**: Stored in `test_set.npy`, containing 100 images with missing information.
- **Generated Images**: Created using a generative model for training purposes.

The objective is to:
1. Use the generative model to create a synthetic dataset for training.
2. Develop a neural network capable of recovering missing portions of the corrupted images.

---

## 📋 Objectives

### Part 1: Dataset Creation
- **Generate Synthetic Images**:
  - Use the provided generative model to create realistic MRI images for training.
  - Save the generated dataset for reuse.
- **Visualize Data**:
  - Display 10 synthetic images and 10 corrupted images from the test set.

### Part 2: Data Preparation
- **Training Dataset**:
  - Create a `PyTorch TensorDataset` with paired corrupted and uncorrupted images.
  - Use a `DataLoader` to provide batches of data for training.
- **Test Dataset**:
  - Create a `PyTorch TensorDataset` with corrupted images only (no labels).
  - Use a `DataLoader` to prepare the test data.
- **Visualize Data**:
  - Display 10 training and 10 test images, along with corresponding labels where applicable.

### Part 3: Neural Network Design
- **Model Architecture**:
  - Design a neural network architecture for image reconstruction.
  - Train the network on the synthetic dataset, ensuring it can generalize to the corrupted test set.

---

## 🛠️ Methodology

1. **Dataset Generation**:
   - Leveraged the provided generative model to create realistic MRI brain images.
   - Corrupted synthetic images to mimic the provided test set.
   - Saved generated data to Google Drive for reuse.

2. **Data Preparation**:
   - Created `TensorDataset` and `DataLoader` objects for training and testing.
   - Visualized data to verify correctness and quality.

3. **Model Training**:
   - Designed and trained a neural network to recover missing image portions.
   - Optimized the network using appropriate loss functions and evaluation metrics.

4. **Evaluation**:
   - Assessed model performance using qualitative visualizations of reconstructed images.

---

## 📂 Repository Structure

- `README.md`: Project overview and description (this file).
- `notebooks/`: Jupyter notebooks for dataset generation, model training, and evaluation.
- `data/`:
  - `test_set.npy`: Provided corrupted images.
  - `generated_images/`: Synthetic training images created using the generative model.
- `models/`: Trained model architectures and weights.
- `results/`: Visualization of reconstructed images and performance metrics.

---

## 🎯 Key Takeaways

This project demonstrates:
- The ability to generate synthetic datasets using generative models.
- Designing and training neural networks for image reconstruction tasks.
- Handling real-world challenges in medical imaging, such as incomplete data.
- Visualizing and evaluating model outputs for qualitative assessment.

---

