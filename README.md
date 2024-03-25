# Breast Cancer Classification using Neural Network

This repository contains Python code for building a simple Neural Network (NN) model to classify breast cancer tumors as either malignant or benign. The project utilizes TensorFlow and Keras libraries for deep learning implementation.

## Contents

- **breast_cancer_classification.ipynb**: Jupyter Notebook containing the Python code for building and training the neural network model.

## Usage

1. Ensure you have Python installed along with necessary libraries such as NumPy, Pandas, Matplotlib, scikit-learn, TensorFlow, and Keras.
2. Clone the repository to your local machine.
3. Open the Jupyter Notebook `breast_cancer_classification.ipynb` using Jupyter Notebook or any compatible environment.
4. Execute the code cells sequentially to load the dataset, preprocess the data, build and train the neural network model, and make predictions.
5. Analyze the model's performance using accuracy and loss metrics plotted over epochs.

## Description

Breast cancer classification is a common machine learning task aimed at distinguishing between malignant and benign tumors based on various features extracted from diagnostic images. In this project, we use the Breast Cancer Wisconsin (Diagnostic) dataset, which is readily available in the scikit-learn library.

The workflow includes:

1. Loading the dataset using scikit-learn's built-in breast cancer dataset.
2. Preprocessing the data, including handling missing values, standardizing features, and splitting the dataset into training and testing sets.
3. Building a neural network model using TensorFlow and Keras. The model architecture consists of an input layer, a hidden layer with rectified linear unit (ReLU) activation, and an output layer with sigmoid activation.
4. Compiling the model with appropriate optimizer, loss function, and evaluation metrics.
5. Training the model on the training data, monitoring its performance on validation data, and visualizing training/validation accuracy and loss over epochs.
6. Evaluating the trained model's performance on the test data and making predictions.
7. Converting prediction probabilities to class labels and interpreting the results.

## Contributor

This project is created by Haichour asma as part of a tutorial on breast cancer classification using neural networks. For any questions or feedback, feel free to reach out.

## References

- [Breast Cancer Wisconsin (Diagnostic) dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
