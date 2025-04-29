# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Step 2: Connect to drive where our dataset stored

from google.colab import drive
drive.mount('/content/drive')



# Step 3: Load the dataset

file_path = '/content/drive/MyDrive/spam.csv'
data = pd.read_csv(file_path, encoding='latin-1')




import re

# Keep only necessary columns and rename them
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Drop rows with missing values
data.dropna(subset=['message'], inplace=True)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing
data['message'] = data['message'].apply(preprocess_text)

# Preview cleaned data
print(data.head())



def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)   # Keep only letters
    text = text.lower()                      # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

# Apply preprocessing
data['message'] = data['message'].apply(preprocess_text)


# 'ham' = 0, 'spam' = 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})


vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(data['message'])
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
