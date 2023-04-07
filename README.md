# Churn Prediction Model

This repository contains a machine learning model for predicting customer churn. The model is trained on a dataset of customer behavior and demographic data, and can be used to predict which customers are most likely to churn in the future.

## Getting Started
To use the churn prediction model, you will need to have Python and several Python packages installed on your system. You can install these packages using pip, by running the following command:
pip install -r requirements.txt
Once you have installed the necessary packages, you can run the churn prediction model by running the churn_prediction_model.py script.
## Dataset
The dataset used to train the churn prediction model is included in the repository as churn_data.csv. This dataset contains information on customer behavior and demographics, as well as a binary flag indicating whether the customer has churned or not.
## Model

The churn prediction model is implemented in Python using the scikit-learn library. The model is a logistic regression model,Random Forest Classifier and Decision Tree Classifier which predicts the probability of customer churn based on the input features.
## Accuracy
* logistic regression: 80%
* Decision Tree Classifier: 78%
* Random Forest Classifier: 86%
## Usage
To use the churn prediction model, simply run the churn_prediction_model.py script. The script will load the dataset, train the model, and output the results of the model on the test set.

You can also modify the script to use your own dataset, by changing the data_file variable to point to your own CSV file.
Requirements

The following Python packages are required to run the churn prediction model:

    scikit-learn
    pandas
    numpy

You can install these packages using pip, by running the following command:

pip install -r requirements.txt
