# Weather Image Recognition using CNN

This project demonstrates the use of a **Convolutional Neural Network (CNN)** to classify weather phenomena based on visual data. The model is trained on a dataset containing 6,862 labeled images across 11 weather categories.

---

## Dataset

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset?resource=download) and is licensed under **CC0 (Public Domain)**. It contains images representing the following weather phenomena:

- **Dew**: 698 images
- **Fog/Smog**: 851 images
- **Frost**: 475 images
- **Glaze**: 639 images
- **Hail**: 591 images
- **Lightning**: 377 images
- **Rain**: 526 images
- **Rainbow**: 232 images
- **Rime**: 1,160 images
- **Sandstorm**: 692 images
- **Snow**: 621 images

**File Size**: Approximately 636.73 MB.

---

## Project Overview

The goal of this project is to build a **multi-class image classification model** using CNNs to identify weather conditions from images. The project involves the following steps:

1. **Data Preprocessing**:
   - Resizing images to a uniform size.
   - Normalizing pixel values to improve model performance.
   - Splitting the dataset into training, validation, and test sets.

2. **Model Architecture**:
   - A CNN model is designed with the following layers:
     - Convolutional layers with ReLU activation.
     - MaxPooling layers for dimensionality reduction.
     - Fully connected layers for classification.
   - Dropout layers are added to prevent overfitting.

3. **Model Training**:
   - The model is trained using the **categorical cross-entropy loss function** and the **Adam optimizer**.
   - Data augmentation techniques (e.g., rotation, flipping, zooming) are applied to improve generalization.

4. **Model Evaluation**:
   - The model's performance is evaluated using metrics such as:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1-Score**
   - A confusion matrix is plotted to visualize classification results.

5. **Visualization**:
   - Training and validation accuracy/loss curves are plotted.
   - Sample predictions are displayed to showcase the model's performance.

---

## Results

- **Accuracy**: The model achieved an accuracy of `XX%` on the test set.
- **Precision**: `YY%`
- **Recall**: `ZZ%`
- **F1-Score**: `AA%`

---

## Dependencies

The following Python libraries are required to run the project:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `keras`
- `scikit-learn`

Install them using the following command:

```bash
pip install numpy pandas matplotlib seaborn tensorflow keras scikit-learn
