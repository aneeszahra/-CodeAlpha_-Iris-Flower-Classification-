# -CodeAlpha_-Iris-Flower-Classification-
Iris Flower Classification
 Overview
This project classifies Iris flowers into one of three species — Setosa, Versicolor, or Virginica — based on their physical measurements (sepal length, sepal width, petal length, and petal width).
It demonstrates a complete Machine Learning pipeline using Python and Scikit-learn, from data exploration to model training, evaluation, and deployment.
Objective
•	Train a machine learning model to classify iris species.
•	Perform Exploratory Data Analysis (EDA).
•	Evaluate and compare multiple models.
•	Optimize the best model using hyperparameter tuning.
•	Save the final model and demonstrate prediction on new samples.
 Key Concepts
•	Classification (Supervised Learning)
•	Data Preprocessing & Standardization
•	Feature Encoding
•	Model Comparison
•	Cross-Validation
•	Hyperparameter Tuning
•	Model Saving & Loading for Inference
Dataset
Source: Kaggle - Iris CSV Dataset
Model Training Workflow
•	Import and clean the dataset
•	Perform data visualization and correlation analysis
•	Preprocess the data (scaling and encoding)
Train and compare models:
•	Random Forest
•	Support Vector Machine (SVM)
•	K-Nearest Neighbors (KNN)
•	Tune the best-performing model with GridSearchCV
•	Evaluate accuracy, precision, recall, and F1-score
•	Save the final model and scaler
•	Test the model on new unseen data
 Results
•	Best Model: Random Forest Classifier
•	Test Accuracy: ~98–100% (depending on random state)
•	Cross-validation Mean Accuracy: ~97–99%
The model correctly distinguishes between all three iris species with high precision and recall.
Dependencies
•	Numpy  
•	pandas
•	Matplotlib 
•	seaborn
•	scikit-learn
•	joblib


