#  Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#  Step 2: Load the dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

#  Step 3: Convert labels to numbers (ham = 0, spam = 1)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

#  Step 4: Separate input (X) and output (y)
X = df['message']
y = df['label_num']

#  Step 5: Split data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Step 6: Convert text into numbers using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

#  Step 7: Train the model using Naive Bayes
model = MultinomialNB()
model.fit(X_train_counts, y_train)

#  Step 8: Make predictions
y_pred = model.predict(X_test_counts)

#  Step 9: Show accuracy
print("\n Accuracy of the model:", accuracy_score(y_test, y_pred))

#  Step 10: Confusion Matrix
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#  Step 11: Classification Report
print("\n Classification Report:\n", classification_report(y_test, y_pred))

#  Step 12: Visualize the Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
