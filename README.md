# Gender Prediction

This repository contains the implementation of a gender prediction model using machine learning techniques. The model predicts the gender of an individual based on a given dataset of with specific  numerical voice features. The project encompasses data preprocessing, model training, evaluation, and deployment.

## Table of Contents
1. [Introduction](#introduction)
2. [Workflow](#workflow)
3. [Dataset](#dataset)
4. [Models Used](#models-used)
5. [Accuracy of Models](#accuracy-of-models)
6. [License](#license)

## Introduction
Gender prediction is a common task in data science, where the goal is to predict the gender of individuals based on features such pitch, frequency, meanfrequency  or other demographic information. This project demonstrates the application of various machine learning algorithms to achieve accurate gender predictions.

## Workflow
The workflow for this project is as follows:
1. **Data Collection**: Gather the dataset containing relevant features for gender prediction.
2. **Data Preprocessing**: Clean and preprocess the data, handling missing values, encoding categorical features, and normalizing the data.
3. **Feature Engineering**: Extract and select the most relevant features for the model.
4. **Model Selection**: Choose appropriate machine learning algorithms for the task.
5. **Model Training**: Train the selected models on the preprocessed dataset.
6. **Model Evaluation**: Evaluate the performance of the models using appropriate metrics.
7. **Hyperparameter Tuning**: Optimize the model parameters to improve accuracy.
8. **Model Deployment**: Deploy the best-performing model for predictions.

## Dataset
The dataset used in this project contains features such as pitch, frequency, meanfrequency and other demographic information. The data is in the voice.csv, and it was preprocessed to ensure quality and consistency. The dataset was split into training and testing sets to evaluate the performance of the models.

## Models Used
Several machine learning models were implemented and compared to determine the best-performing model for gender prediction. The models used include:
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest**
4. **Gaussian Process Classifier**
5. **Decision Tree**
6. **k-Nearest Neighbors**

   

## Accuracy of Models
The performance of each model was evaluated using metrics such as accuracy, precision, recall, and F1-score. The results are summarized below:
- **Logistic Regression**: Accuracy - 0.910620
- **Support Vector Machine (SVM)**: Accuracy - 0.969506
- **Random Forest**: Accuracy - 0.974763
- **Gaussian Process Classifier**: Accuracy - 0.730810
- **Decision Tree**: Accuracy -0.973712
-  **k-Nearest Neighbors**: Accuracy -0.716088

Based on these evaluations, the Neural Networks model demonstrated the highest accuracy for gender prediction.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
