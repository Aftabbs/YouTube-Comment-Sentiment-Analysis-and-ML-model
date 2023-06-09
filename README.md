# YouTube Comment Sentiment Analysis
![image](https://github.com/Aftabbs/YouTube-Comment-Sentiment-Analysis-and-ML-model/assets/112916888/dd234f95-a460-4353-96c6-c86d3d8f2c20)

# Project Overview
This project focuses on performing sentiment analysis on YouTube comments using the Google API Key for comment scraping. The objective is to analyze the sentiment polarity of the comments and build a machine learning model to predict the sentiment of new comments.
![image](https://github.com/Aftabbs/YouTube-Comment-Sentiment-Analysis-and-ML-model/assets/112916888/6c443833-f4a7-4763-a4f7-dc022d450434)

# Dataset
The dataset consists of YouTube comments scraped using the Google API Key,Which is a Automated Process,We Can Scrape Any Kind of YT Videos Comments Just By Providing VideoID . The comments are stored in a DataFrame containing the comment text and corresponding sentiment labels. The dataset may be imbalanced, with an unequal distribution of positive, negative, and neutral sentiments.

# Preprocessing Steps
* Data Collection: YouTube comments are scraped using the Google API Key, and the scraped comments are stored in a DataFrame.
* Sentiment Analysis: The NLTK library is used to perform sentiment analysis on the original comments, providing insights into the initial sentiment distribution.
![image](https://github.com/Aftabbs/YouTube-Comment-Sentiment-Analysis-and-ML-model/assets/112916888/c5a431dd-2603-4177-b913-84581e2d0f60)
![image](https://github.com/Aftabbs/YouTube-Comment-Sentiment-Analysis-and-ML-model/assets/112916888/ad949ca3-c9d9-4453-9e85-9f6cc41d1260)
* Data Preprocessing: The comments undergo preprocessing steps such as text cleaning, including removing special characters and numbers, lowercasing the text, and  removing stopwords.
* Stemming: The comments are processed using stemming techniques to reduce words to their base or root form, which helps in reducing the dimensionality of the data.
* Balancing the Dataset: The imbalanced dataset is balanced using techniques such as oversampling or undersampling to ensure equal representation of positive, negative, and neutral sentiments.

![image](https://github.com/Aftabbs/YouTube-Comment-Sentiment-Analysis-and-ML-model/assets/112916888/214a7836-421c-4a89-98f8-c0d2ecf5b330)

* Label Encoding: The sentiment labels (positive, negative, neutral) are encoded using LabelEncoder from scikit-learn, converting them into numerical representations for model training.

# Machine Learning Model
A Gaussian Naive Bayes (GaussianNB) model is chosen for sentiment classification. This model assumes that the features (word counts) are independent of each other and follow a Gaussian distribution. It is a probabilistic classifier that predicts the sentiment of comments based on the occurrence of words.

# Feature Vectorization
The comments are vectorized using CountVectorizer from scikit-learn, which converts the text data into numerical feature vectors. This step transforms the comments into a matrix representation, where each row represents a comment, and each column represents a unique word in the entire corpus.

# Model Training and Evaluation
The GaussianNB model is trained on the vectorized comments. The model's performance is evaluated using various evaluation metrics, including the confusion matrix and accuracy score. The confusion matrix provides insights into the true positives, true negatives, false positives, and false negatives, helping to assess the model's performance on different sentiment classes.

# Additional Enhancements
To further improve the project, consider the following enhancements:

* Advanced Text Preprocessing: Explore advanced techniques for text preprocessing, such as lemmatization, spell checking, or handling emoticons and abbreviations commonly used in social media comments.
* Feature Engineering: Experiment with additional features derived from the comments, such as comment length, presence of emojis or hashtags, or sentiment scores from other sentiment analysis libraries.
* Advanced Machine Learning Models: Explore other machine learning models such as Support Vector Machines (SVM), Random Forests, or deep learning models like Recurrent Neural Networks (RNN) or Transformers for sentiment classification.
* Model Fine-tuning: Optimize the model's hyperparameters using techniques like GridSearchCV to find the best combination of parameters for improved performance.
* Deployment: Consider deploying the trained model as a web application or API, allowing users to input new comments and receive sentiment predictions in real-time.

# Industry Use Case
* Sentiment analysis of YouTube comments has several industry applications, including:
Brand Reputation Management: Analyzing sentiments of user comments about a brand or product to understand customer sentiment and identify areas for improvement.
Market Research: Analyzing sentiments of YouTube comments

# Enjoy The Projecting By Considering Multiple Video Availabel on YT to Analyse,Interpret,Evaluate and Gain Info





