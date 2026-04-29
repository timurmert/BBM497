import numpy as np
import pandas as pd
import gensim.downloader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


model = gensim.downloader.load("word2vec-google-news-300")
EMBEDDING_DIM = 300


def document_vector(text):
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


class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.fc1(x)
        h = self.relu(h)
        out = self.fc2(h)
        return out


def train_classifier(X_train, y_train,
                     hidden_dim=64,
                     learning_rate=0.001,
                     epochs=10,
                     batch_size=32):
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    clf = SentimentClassifier(input_dim, hidden_dim)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(clf.parameters(), lr=learning_rate)

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
    clf.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        logits = clf(X_tensor)
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).long().view(-1).numpy()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics
