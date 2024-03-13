import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('Churn_Modelling.csv')

# Preprocessing
# Drop irrelevant columns like 'RowNumber', 'CustomerId', and 'Surname'
data.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

# Splitting data into features and target
X = data.drop(columns=['Exited'])  # Features
y = data['Exited']  # Target variable

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Evaluation for {name}:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix for {name}:")
    print(confusion_matrix(y_test, y_pred))
