import numpy as np
import pandas as pd
import gensim.downloader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Pre-trained embedding model (do not modify)
model = gensim.downloader.load("word2vec-google-news-300")
EMBEDDING_DIM = 300


def document_vector(text):
    """
    Convert text into a single vector by averaging word embeddings.

    Requirements:
    - Tokenize the text based on whitespace.
    - Ignore out-of-vocabulary (OOV) words.
    - If all words are OOV, return a zero vector with shape (300,).

    Args:
        text (str): input review text

    Returns:
        np.ndarray: document vector with shape (300,)
    """
    tokens = text.split()

    valid_vectors = []
    for word in tokens:
        if word in model:
            valid_vectors.append(model[word])

    if len(valid_vectors) == 0:
        return np.zeros(EMBEDDING_DIM)

    total = np.zeros(EMBEDDING_DIM)
    for vec in valid_vectors:
        total = total + vec

    avg_vector = total / len(valid_vectors)
    return avg_vector


def prepare_classification_data(file_path):
    """
    Load the IMDB dataset and convert it into feature and label arrays.

    Expected CSV format:
        review,sentiment

    Label mapping:
        positive -> 1
        negative -> 0

    Args:
        file_path (str): path to the CSV file

    Returns:
        X (np.ndarray): feature matrix with shape (N, 300)
        y (np.ndarray): label vector with shape (N,)
    """
    df = pd.read_csv(file_path)

    reviews = df["review"].tolist()
    sentiments = df["sentiment"].tolist()

    X_list = []
    y_list = []
    for i in range(len(reviews)):
        review_text = str(reviews[i])
        label_text = sentiments[i]

        vec = document_vector(review_text)
        X_list.append(vec)

        if label_text == "positive":
            y_list.append(1)
        else:
            y_list.append(0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def train_classifier(X_train, y_train,
                     hidden_dim=64,
                     learning_rate=0.001,
                     epochs=10,
                     batch_size=32):
    """
    Train a simple neural network classifier using document vectors.

    Requirements:
    - Use a deep learning framework such as PyTorch.
    - The model must have:
        * an input layer matching the embedding dimension,
        * at least one hidden layer,
        * an output layer for binary classification.
    - The function must return the trained model.

    Args:
        X_train (np.ndarray): training feature matrix
        y_train (np.ndarray): training labels

    Returns:
        torch.nn.Module: trained neural network classifier
    """
    class SentimentClassifier(nn.Module):
        def __init__(self, input_dim, hidden):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.fc1(x)
            h = self.relu(h)
            out = self.fc2(h)
            return out

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    clf = SentimentClassifier(input_dim, hidden_dim)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)

    clf.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = clf(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return clf

def evaluate_classifier(clf, X_test, y_test):
    """
    Evaluate the trained classifier.

    Requirements:
    - Predict labels for X_test.
    - Compute accuracy, precision, recall, and F1-score.

    Args:
        clf (torch.nn.Module): trained classifier
        X_test (np.ndarray): test feature matrix
        y_test (np.ndarray): test labels

    Returns:
        dict: {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float
        }
    """
    clf.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        logits = clf(X_tensor)
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).long().view(-1).numpy()

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(y_test)):
        true_label = int(y_test[i])
        pred_label = int(predictions[i])
        if pred_label == 1 and true_label == 1:
            tp = tp + 1
        elif pred_label == 1 and true_label == 0:
            fp = fp + 1
        elif pred_label == 0 and true_label == 1:
            fn = fn + 1
        else:
            tn = tn + 1

    total = tp + fp + fn + tn
    if total == 0:
        accuracy = 0.0
    else:
        accuracy = (tp + tn) / total

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
