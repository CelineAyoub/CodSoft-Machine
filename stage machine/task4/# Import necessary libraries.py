import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Preprocessing
# Drop unnecessary columns and rename columns for clarity
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Convert labels to binary (0 for legitimate, 1 for spam)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Splitting data into features and target
X = data['text']
y = data['label']

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = nb_classifier.predict(X_test)
print("Evaluation for Naive Bayes Classifier:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred)))

# Example of how to make predictions on new SMS messages
new_messages = [
    "Congratulations! You've won a free cruise. Claim your prize now!",
    "Hi, just wanted to remind you about our meeting tomorrow."
]
new_messages_tfidf = tfidf_vectorizer.transform(new_messages)
predictions = nb_classifier.predict(new_messages_tfidf)
for message, prediction in zip(new_messages, predictions):
    print(f"Message: {message}")
    print("Prediction: Spam" if prediction == 1 else "Prediction: Legitimate")
