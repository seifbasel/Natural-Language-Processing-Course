import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from minisom import MiniSom
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

# List of columns to remove
columns_to_remove = ['number of reviews', 'Title', 'Clothing ID', 'Age', 'Division Name', 'Department Name', 'Class Name']

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

        # Stem tokens using the Snowball stemmer
        stemmer = SnowballStemmer(language='english')
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        # Join tokens back into text
        processed_text = ' '.join(stemmed_tokens)
        return processed_text
    else:
        return ""  # Return an empty string if text is not a string

# Apply preprocessing to the Review Text column
data['Review Text'] = data['Review Text'].apply(preprocess_text)

# Define features (X) and target (y)
X = data['Review Text']
y = data['Rating'].apply(lambda x: 1 if x > 3 else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Multilayer Perceptron (MLP) model with different hidden layer sizes
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, alpha=1e-4, solver='adam', verbose=10, random_state=42, tol=0.0001)
mlp_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred_mlp = mlp_model.predict(X_test_tfidf)

# Evaluate the MLP model
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("MLP Model Accuracy:", int(accuracy_mlp * 100), '%')


# Compute confusion matrix for MLP model
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

# Visualize confusion matrix for MLP model
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - MLP Model')
plt.show()



# Train an Adaptive Linear Neuron (Adaline) model
adaline_model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
adaline_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred_adaline = adaline_model.predict(X_test_tfidf)

# Evaluate the Adaline model
accuracy_adaline = accuracy_score(y_test, y_pred_adaline)
print("Adaline Model Accuracy:", int(accuracy_adaline * 100), '%')


# Compute confusion matrix for Adaline model
cm_adaline = confusion_matrix(y_test, y_pred_adaline)

# Visualize confusion matrix for Adaline model
plt.figure(figsize=(8, 6))
sns.heatmap(cm_adaline, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Adaline Model')
plt.show()


# Convert sparse matrix to dense matrix
X_train_dense = X_train_tfidf.toarray()

# Define SOM parameters
som_cols = 10
som_rows = 10
input_len = X_train_dense.shape[1]  # Number of features from TF-IDF

# Initialize the SOM
som = MiniSom(som_cols, som_rows, input_len, sigma=0.5, learning_rate=0.5, random_seed=42)

# Train the SOM
som.train_random(X_train_dense, 1000)  # Adjust the number of iterations as needed

# Assign samples to their closest neurons
winner_coordinates = np.array([som.winner(x) for x in X_train_dense]).T

# Create a dictionary to store the lists of documents for each neuron
cluster_indices = {}
for i, j in enumerate(winner_coordinates[0]):
    if j not in cluster_indices:
        cluster_indices[j] = [i]
    else:
        cluster_indices[j].append(i)

# Use Hebbian-like rule to assign labels to clusters
cluster_labels = {}
for cluster, indices in cluster_indices.items():
    # Count positive and negative labels
    positive_count = sum(y_train.iloc[indices])
    negative_count = len(indices) - positive_count
    
    # Assign label based on majority
    cluster_labels[cluster] = 1 if positive_count > negative_count else 0


class HebbianNeuralNetwork:
    def __init__(self, input_size, learning_rate=0.1):
        self.input_size = input_size
        self.weights = np.zeros((input_size, input_size))  # Initialize weights as a square matrix
        self.learning_rate = learning_rate

    def train(self, X):
        for x in X:
            self.weights += self.learning_rate * np.outer(x, x)

    def predict(self, X):
        activations = np.dot(X, self.weights)
        return np.where(activations >= 0, 1, 0)


# Convert text data to binary representation
def text_to_binary(text, vocabulary):
    binary = np.zeros(len(vocabulary))
    for word in text.split():
        if word in vocabulary:
            binary[vocabulary.index(word)] = 1
    return binary

# Create vocabulary from training data
vocabulary = set()
for text in X_train:
    for word in text.split():
        vocabulary.add(word)
vocabulary = list(vocabulary)

# Convert text data to binary features
X_train_binary = np.array([text_to_binary(text, vocabulary) for text in X_train])
X_test_binary = np.array([text_to_binary(text, vocabulary) for text in X_test])

# Train Hebbian Neural Network
hnn = HebbianNeuralNetwork(len(vocabulary))
hnn.train(X_train_binary)

# Make predictions
y_pred_hnn = hnn.predict(X_test_binary)

# Evaluate the HNN model
accuracy_hnn = accuracy_score(y_test, y_pred_hnn)
print("Hebbian Neural Network Accuracy:", int(accuracy_hnn * 100), '%')


# Compute confusion matrix for Hebbian Neural Network model
cm_hnn = confusion_matrix(y_test, y_pred_hnn)

# Visualize confusion matrix for Hebbian Neural Network model
plt.figure(figsize=(8, 6))
sns.heatmap(cm_hnn, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Hebbian Neural Network Model')
plt.show()
