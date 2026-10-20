# Heart Disease Prediction Using Machine Learning

A machine learning project that explores the use of classification algorithms to predict the presence or absence of heart disease based on patient health attributes.

The project compares Logistic Regression, Random Forest, and XGBoost and integrates the selected model into an interactive web application built with Streamlit.

## About the Project

The goal of this project is to understand how machine learning can be applied to healthcare-related datasets. It covers the complete workflow, from data preprocessing and exploratory analysis to model training, evaluation, and application development.

Users can enter patient attributes through the Streamlit interface and view the classification produced by the trained model.

This project is intended for educational purposes and is not a clinically validated diagnostic tool.

## Objectives

* Explore and preprocess the Heart Disease dataset.
* Perform exploratory data analysis to understand patterns in the data.
* Train and evaluate multiple machine learning classification models.
* Compare models using appropriate evaluation metrics.
* Analyze feature importance.
* Build an interactive web application using Streamlit.
* Prepare the application for deployment.

## Dataset

**Source:** UCI Machine Learning Repository

**Dataset:** [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

This project uses the processed Cleveland dataset, which contains 13 input features and one target variable.

### Features

The dataset includes the following patient attributes:

* Age and sex
* Chest pain type
* Resting blood pressure
* Serum cholesterol
* Fasting blood sugar
* Resting electrocardiogram results
* Maximum heart rate achieved
* Exercise-induced angina
* ST depression and ST segment slope
* Number of major vessels
* Thalassemia-related test category

### Target Variable

The original dataset contains target values from 0 to 4. For this project, the target is converted into a binary classification variable:

* `0`: No heart disease indicated in the dataset.
* `1`: Heart disease indicated in the dataset.

## Technologies Used

| Technology   | Purpose                                            |
| ------------ | -------------------------------------------------- |
| Python       | Core programming language                          |
| Pandas       | Data manipulation and analysis                     |
| NumPy        | Numerical operations                               |
| Matplotlib   | Data visualization                                 |
| Seaborn      | Statistical visualization                          |
| Scikit-learn | Data preprocessing, model training, and evaluation |
| XGBoost      | Gradient boosting classification                   |
| Joblib       | Saving and loading trained models                  |
| Streamlit    | Building the interactive web application           |

## Machine Learning Models

The project explores three classification algorithms:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

The models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

Stratified cross-validation is also used to obtain a more reliable view of model performance across different data splits.

## Project Workflow

1. Dataset collection and loading
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Correlation analysis
5. Feature preparation
6. Train-test splitting
7. Model training
8. Model evaluation
9. Cross-validation
10. Model comparison
11. Saving the selected model
12. Developing the Streamlit application

## Project Structure

```text
Heart-Disease-Prediction/
|
|-- dataset/
|   |-- processed.cleveland.data
|
|-- Heart_Disease_Prediction.ipynb
|-- app.py
|-- heart_disease_model.pkl
|-- requirements.txt
|-- README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace the placeholder with the URL of your GitHub repository.

### 2. Navigate to the project directory

```bash
cd Heart-Disease-Prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local address displayed in the terminal, usually `http://localhost:8501`.

## Application Features

* Interactive form for entering patient attributes
* Prediction using a trained machine learning model
* Display of the predicted class
* Model-estimated probability
* Simple interface built with Streamlit

## Future Improvements

* Hyperparameter tuning to improve model performance
* More extensive model validation
* Model explainability using SHAP
* Improved input validation and error handling
* Additional automated testing
* Cloud deployment

## Limitations

The dataset is relatively small, and performance estimates may vary depending on the train-test split. Model predictions have not been clinically validated and should not be interpreted as medical diagnoses or individual clinical risk estimates.

Feature importance describes predictive relationships within the model; it does not establish medical causation.

## References

* [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* [Streamlit Documentation](https://docs.streamlit.io/)

## Author

Tanaji Shirish Bodadkar

Python | Machine Learning | Data Science

```

**Next step:** Save this as `README.md` in your project folder. Replace `Your Name` and `YOUR_GITHUB_REPOSITORY_URL` with your actual details when you're ready to publish the project.
```
