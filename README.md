# Telco-Customer-Churn
The aims of this project is to predict the customer churn (i.e., whether a consumer is going to stop using the service) using the Telco Customer Churn Dataset from Kaggle. 

## Objective
The goal of this project is to:
* Build an accurate classification model to predict churn
* Perform EDA to understand key features of Churn.
* Apply ML techniques to achieve remarkable prediction
* Visualize for beter interpretability

## Project Workflow
### Data Preprocessing
* Missing values: Handled missign or null values in Dataset
* Encoding: Converted values into binary and such values into machine readable values for model training/evaluation.
* Data Cleaning: Cleaned inconsistent values
* Scaling: Scaled numerical featuers for better model performance.
### EDA
* Identified trends and patterns in customer demographics, service usage, and account tenure.
* Visualized relationships between customer attributes and churn using:
  * Histograms, bar charts, and boxplots.
  * Correlation heatmaps to identify key features.
 
### Model Training and Evaluation
Applied multiple machine learning models to identify the best-performing algorithm for churn prediction:
  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting
  * K-Nearest Neighbors (KNN)
  * Naive Bayes
  * Support Vector Machine (SVM)
Model Tuning:
* Random Forest: Optimized hyperparameters (max_depth, min_samples_split, n_estimators) to improve accuracy and ROC-AUC.
* Logistic Regression: Increased the maximum number of iterations to ensure model convergence.

### Model Comparison
Evaluated models using metrics such as:
* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
Visualized performance using confusion matrices and ROC curves for better comparison

### Results
The Random Forest model achieved the best overall performance with a high ROC-AUC score and balanced precision-recall metrics.
Logistic Regression was also fine-tuned to perform competitively with interpretable results.

### Tools and Technologies Used
* Programming Language: Python
* Libraries:
  Data Analysis: Pandas, NumPy
  Visualization: Matplotlib, Seaborn
  Machine Learning: Scikit-learn, XGBoost
Dataset: Telco Customer Churn Dataset (Kaggle)

### Conclusion
The project successfully developed and evaluated machine learning models for predicting customer churn. By using feature engineering, model tuning, and evaluation techniques, to identify key drivers of churn and build a reliable predictive framework.
