# Sentiment Analysis on Twitter Data

## Project Overview
This project performs sentiment analysis on Twitter data using PySpark. It involves data preprocessing, feature engineering, machine learning classification, and visualization to determine the sentiment of tweets as Positive, Negative, or Neutral.

## Technologies Used
- **Programming Language:** Python
- **Big Data Processing:** Apache Spark (PySpark - Spark SQL, MLlib)
- **Natural Language Processing:** TextBlob
- **Data Visualization:** Matplotlib, Seaborn
- **Language Detection:** Langdetect
- **Development Environment:** Jupyter Notebook

## Dataset
The dataset used is a CSV file containing tweets with their respective sentiment labels. Ensure the dataset path is correctly set in the script before running the project.

## Features Implemented
### 1. Data Preprocessing
- Removal of retweets
- Dropping duplicate tweets and unnecessary columns
- Filtering relevant tweets based on predefined conditions
- Tokenization and stop-word removal
- Emoji and URL replacement
- Removing special characters and retaining only alphanumeric content

### 2. Sentiment Analysis
- Sentiment classification using TextBlob
- Tweets labeled as **Positive**, **Neutral**, or **Negative**

### 3. Machine Learning Model
- **Text Vectorization:** Using TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classification Model:** Logistic Regression for sentiment classification
- **Model Evaluation:** Accuracy calculation and confusion matrix visualization

### 4. Regression Analysis
- **Model Used:** Linear Regression to analyze sentiment trends
- **Evaluation Metrics:** RMSE (Root Mean Squared Error) and R² (R-Squared)
- **Visualization:** Scatter plot to represent sentiment trends

### 5. Language Detection
- Detects the language of tweets using `langdetect`
- Filters out non-English tweets for sentiment analysis

## Installation
To run this project, install the required dependencies:
```sh
pip install pyspark textblob emoji matplotlib seaborn langdetect
```
Ensure Apache Spark is properly configured in your environment.

## How to Run
1. Open Jupyter Notebook or any Python script editor.
2. Load the dataset by specifying the correct file path.
3. Execute the script to preprocess data, train the model, and generate results.
4. View outputs such as sentiment classification, regression analysis, and visualizations.

## Results
- Sentiment analysis model accuracy displayed after training.
- Confusion matrix plotted using Seaborn.
- Scatter plot visualizing sentiment trends based on linear regression.

## Future Enhancements
- Implement deep learning models like LSTMs and BERT for improved accuracy.
- Extend dataset to include multilingual sentiment analysis.
- Deploy the model as a web application or API for real-time sentiment classification.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

