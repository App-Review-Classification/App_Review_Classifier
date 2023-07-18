# App_Review_Classifier
App reviews play a crucial role in understanding users' opinions and sentiments towards an application. The App Review Classifier project leverages machine learning techniques to automatically analyze and classify app reviews in five categories. By accurately identifying the sentiment behind user reviews, this project can provide valuable insights to app developers and businesses for enhancing their products and services.

## Introduction
In this project, we aim to build a sentiment classifier for app reviews. The code provided performs the following steps:

Data Extraction: Reads the app review data from CSV files.

Data Preprocessing: Preprocesses the reviews by converting them to lowercase, removing punctuation, stopwords, rare words, and applying stemming. It also converts chat words and emojis.

Model Training: Uses a Random Forest Classifier to train a sentiment classification model based on the preprocessed reviews.

Model Evaluation: Evaluates the trained model's performance by predicting the sentiment labels for a test dataset and calculating the accuracy.

## Requirements
The code is implemented in Python and requires the following dependencies:

pandas
numpy
nltk
scikit-learn

## Data preprocessing
The preprocessing steps applied to the app review data include:

Converting the Reviews to Lowercase:

1. All the text in the reviews is converted to lowercase. This step ensures that the model treats words in uppercase and lowercase as the same, avoiding any discrepancies due to case sensitivity.
Removing Punctuation:

2. Punctuation marks such as periods, commas, and exclamation marks are removed from the reviews. This step helps in reducing noise and focusing on the essential words in the text.
Removing Stopwords:

3. Stopwords are common words that do not carry much meaning in the context of sentiment analysis. Examples of stopwords include "the," "is," "and," and "but." These words are removed from the reviews to reduce noise and improve the model's ability to focus on more meaningful words.
Removing Rare Words:

4. Rare words are words that occur infrequently in the dataset. They might be typos, misspellings, or words specific to a few reviews. Removing rare words helps in reducing noise and improving the model's ability to generalize well to unseen data.
Applying Stemming:

5. Stemming is the process of reducing words to their base or root form. It helps in reducing the dimensionality of the data by treating different word forms (e.g., "running," "runs," "ran") as the same. This step ensures that the model can generalize well by considering different word forms as equivalent.
Converting Chat Words and Emojis:

6. Chat words and emojis are common in informal text such as app reviews. Converting chat words to their expanded forms and replacing emojis with their textual representations helps in standardizing the text and making it more suitable for sentiment analysis. This step ensures that the model can effectively interpret the sentiment conveyed by these expressions.

By applying these preprocessing steps, the app review data is transformed into a clean and standardized format, making it more suitable for training a sentiment classification model. These steps help in reducing noise, improving model performance, and ensuring that the model captures the essential sentiment information from the reviews.

## Model training
The sentiment classification model is trained using a Random Forest Classifier. Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. Each tree in the forest is trained on a subset of the data, and the final prediction is made by averaging the predictions of all the individual trees.

 The preprocessed app review data is divided into training and test sets. The training set is used to train the model by extracting features from the preprocessed text data using TF-IDF vectorization and scaling the features. The Random Forest Classifier is then trained on the scaled features, learning to classify app reviews based on sentiment labels. The trained model is evaluated using the test set to measure its accuracy in predicting sentiment labels for app reviews.
