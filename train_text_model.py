# train_text_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load the symptom dataset (ensure the CSV file is in the right path)
df = pd.read_csv('skin_disease_symptoms.csv')

# Features (symptoms) and target (disease)
X = df['Symptoms']
y = df['Disease']

# Text vectorization using TF-IDF (converts text into numerical features)
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')  # Convert text data to feature vectors
X_tfidf = tfidf.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model and TF-IDF vectorizer
with open('text_symptom_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

print("Model and TF-IDF vectorizer saved!")
