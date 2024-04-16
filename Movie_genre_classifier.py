import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load training data
train_data = pd.read_csv("C:/Users/Admin/Desktop/Codeway/data/train_data.csv", sep=":::", header=None, names=["ID", "TITLE", "GENRE", "DESCRIPTION"], engine="python", index_col=False)

# Preprocess training data
X = train_data['DESCRIPTION']
y = train_data['GENRE']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Train Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tfidf, y_train)

# Load test data
test_data = pd.read_csv("C:/Users/Admin/Desktop/Codeway/data/test_data.csv", sep=":::", header=None, names=["ID", "TITLE", "DESCRIPTION"], engine="python", index_col=False)

# Preprocess test data
X_test = test_data['DESCRIPTION']
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Make predictions on test data
test_predictions = naive_bayes_classifier.predict(X_test_tfidf)

# Add predictions to test data DataFrame
test_data['PREDICTED_GENRE'] = test_predictions

# Save predictions to CSV file
test_data.to_csv("C:/Users/Admin/Desktop/Codeway/data/test_data_with_predictions(1).csv", index=False)
