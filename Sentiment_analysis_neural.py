import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

# List of columns to remove
columns_to_remove = ['number of reviews','Title','Clothing ID', 'Age', 'Division Name', 'Department Name', 'Class Name']

# Drop the specified columns
data.drop(columns_to_remove, axis=1, inplace=True)

# Preprocessing the Review Text column
def preprocess_text(text):
    # Check if text is a string
    if isinstance(text, str):
        # Convert text to lowercase
        text = text.lower()

        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]

        stemmer = SnowballStemmer(language='english')  # Initialize Snowball stemmer
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]  # Stem tokens

        # Join tokens back into text
        processed_text = ' '.join(stemmed_tokens)
        return processed_text
    else:
        return ""  # Return an empty string if text is not a string

# Apply preprocessing to the Review Text column
data['Review Text'] = data['Review Text'].apply(preprocess_text)

# Define features (X) and target (y)
X = data['Review Text']
y = data['Rating'].apply(lambda x: 1 if x >= 3 else 0)  # Convert ratings to binary sentiment labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=2000)  # You can adjust the max_features parameter as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

# Train a perceptron model
perceptron_model = Perceptron()
perceptron_model.fit(X_train_tfidf, y_train)

# Train an MLP feedforward neural network
mlp_feedforward_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, solver='adam')
mlp_feedforward_model.fit(X_train_tfidf, y_train)

# Train an MLP with backpropagation
mlp_backpropagation_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
mlp_backpropagation_model.fit(X_train_tfidf, y_train)

# Train an Adaline model
adaline_model = SGDClassifier(loss='perceptron', eta0=0.1, learning_rate='constant', penalty=None)
adaline_model.fit(X_train_tfidf, y_train)


import numpy as np

class HebbianLearning:
    def __init__(self, input_size):
        self.weights = np.zeros((input_size, input_size))

    def train(self, X):
        for x in X:
            self.weights += np.outer(x, x)

    def predict(self, X):
        activations = np.dot(X, self.weights)
        return np.where(activations >= 0, 1, 0)

# Example usage:
X_train_hebbian = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y_train_hebbian = np.array([1, 0, 0, 0])

hebbian_model = HebbianLearning(input_size=2)
hebbian_model.train(X_train_hebbian)

X_test_hebbian = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
predictions_hebbian = hebbian_model.predict(X_test_hebbian)



# Make predictions on the test set
y_pred_logistic = logistic_model.predict(X_test_tfidf)
y_pred_perceptron = perceptron_model.predict(X_test_tfidf)
y_pred_mlp_feedforward = mlp_feedforward_model.predict(X_test_tfidf)
y_pred_mlp_backpropagation = mlp_backpropagation_model.predict(X_test_tfidf)
y_pred_adaline = adaline_model.predict(X_test_tfidf)

# Evaluate the models
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
accuracy_mlp_feedforward = accuracy_score(y_test, y_pred_mlp_feedforward)
accuracy_mlp_backpropagation = accuracy_score(y_test, y_pred_mlp_backpropagation)
accuracy_adaline = accuracy_score(y_test, y_pred_adaline)





# Confusion Matrix and Classification Report
print("Logistic Regression Model:")
print("Accuracy:", int(accuracy_logistic * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))
print("Classification Report:")
print(classification_report(y_test, y_pred_logistic))

print("\nPerceptron Model:")
print("Accuracy:", int(accuracy_perceptron * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_perceptron))
print("Classification Report:")
print(classification_report(y_test, y_pred_perceptron))

print("\nMLP Feedforward Model:")
print("Accuracy:", int(accuracy_mlp_feedforward * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mlp_feedforward))
print("Classification Report:")
print(classification_report(y_test, y_pred_mlp_feedforward))

print("\nMLP Backpropagation Model:")
print("Accuracy:", int(accuracy_mlp_backpropagation * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mlp_backpropagation))
print("Classification Report:")
print(classification_report(y_test, y_pred_mlp_backpropagation))

print("\nAdaline Model:")
print("Accuracy:", int(accuracy_adaline * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_adaline))
print("Classification Report:")
print(classification_report(y_test, y_pred_adaline))

print("Predictions using Hebbian Model:")
for inputs, prediction in zip(X_test_hebbian, predictions_hebbian):
    print("Inputs:", inputs)
    print("Predicted Output:", prediction)

# Visualize Confusion Matrices
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')


models = ["Logistic Regression", "Perceptron", "MLP Feedforward", "MLP Backpropagation", "Adaline"]
for i, y_pred in enumerate([y_pred_logistic, y_pred_perceptron, y_pred_mlp_feedforward, y_pred_mlp_backpropagation, y_pred_adaline]):
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels=["Negative", "Positive"])
    plt.title(f"Confusion Matrix - {models[i]} Model")
    plt.show()


new_reviews = [
    "This dress is amazing I love it",
    "very good quality",
    "good fit",
    "nice product",
    "loved it",
    "The quality is very poor.",
    "the fit is bad",
    "i dont like it",
    "not worth money",
    "bad quality",
]

# Print the predictions using all models
print("\nPredictions using Logistic Regression Model:")
for review, prediction in zip(new_reviews, y_pred_logistic):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()

print("\nPredictions using Perceptron Model:")
for review, prediction in zip(new_reviews, y_pred_perceptron):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()

# print("\nPredictions using MLP Model:")
# for review, prediction in zip(new_reviews, predictions_mlp):
#     sentiment = "Positive" if prediction == 1 else "Negative"
#     print("Review:", review)
#     print("Predicted Sentiment:", sentiment)
#     print()
    
    
print("\nPredictions using MLP Feedforward Model:")
for review, prediction in zip(new_reviews, y_pred_mlp_feedforward):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()

print("\nPredictions using MLP Backpropagation Model:")
for review, prediction in zip(new_reviews, y_pred_mlp_backpropagation):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()    


print("\nPredictions using Adaline Model:")
for review, prediction in zip(new_reviews, y_pred_adaline):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()

print("\nPredictions using Hebbian Model:")
for inputs, prediction in zip(X_test_hebbian, predictions_hebbian):
    print("Inputs:", inputs)
    print("Predicted Output:", prediction)