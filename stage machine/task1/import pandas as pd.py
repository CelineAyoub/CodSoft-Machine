import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('genre_classification_dataset.csv')

# Preprocessing
# For simplicity, let's drop rows with missing values
data.dropna(inplace=True)

# Splitting data into features and target
X = data['plot_summary']
y = data['genre']

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC()
}

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Evaluation for {name}:")
    print(classification_report(y_test, y_pred))

# Choose the best performing classifier and fine-tune if necessary
# Here, you can choose based on evaluation metrics like accuracy, precision, recall, F1-score, etc.

# Make predictions on new data if needed
