import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

def extract_features(sentence, index):

    word = sentence[index]

    has_digit = False
    for ch in word:
        if ch.isdigit():
            has_digit = True
            break

    has_upper = False
    for ch in word:
        if ch.isupper():
            has_upper = True
            break

    features = {}

    features['word'] = word.lower()

    features['is_first'] = (index == 0)
    features['is_last'] = (index == len(sentence) - 1)

    features['is_capitalized'] = word[0].isupper()
    features['is_all_caps'] = word.isupper()
    features['is_all_lower'] = word.islower()
    features['has_digit'] = has_digit
    features['has_upper'] = has_upper
    features['has_hyphen'] = '-' in word
    features['has_period'] = '.' in word

    features['length'] = len(word)

    features['prefix_1'] = word[:1].lower()
    features['prefix_2'] = word[:2].lower()
    features['prefix_3'] = word[:3].lower()

    features['suffix_1'] = word[-1:].lower()
    features['suffix_2'] = word[-2:].lower()
    features['suffix_3'] = word[-3:].lower()

    if index > 0:
        prev_word = sentence[index - 1]
        features['prev_word'] = prev_word.lower()
        features['prev_suffix_2'] = prev_word[-2:].lower()
        features['prev_suffix_3'] = prev_word[-3:].lower()
        features['prev_is_capitalized'] = prev_word[0].isupper()
    else:
        features['prev_word'] = '<START>'
        features['prev_suffix_2'] = '<START>'
        features['prev_suffix_3'] = '<START>'
        features['prev_is_capitalized'] = False

    if index > 1:
        prev2_word = sentence[index - 2]
        features['prev2_word'] = prev2_word.lower()
    else:
        features['prev2_word'] = '<START2>'

    if index < len(sentence) - 1:
        next_word = sentence[index + 1]
        features['next_word'] = next_word.lower()
        features['next_suffix_2'] = next_word[-2:].lower()
        features['next_suffix_3'] = next_word[-3:].lower()
        features['next_is_capitalized'] = next_word[0].isupper()
    else:
        features['next_word'] = '<END>'
        features['next_suffix_2'] = '<END>'
        features['next_suffix_3'] = '<END>'
        features['next_is_capitalized'] = False

    if index < len(sentence) - 2:
        next2_word = sentence[index + 2]
        features['next2_word'] = next2_word.lower()
    else:
        features['next2_word'] = '<END2>'

    return features

def main():
    tagged_sents = list(brown.tagged_sents(tagset='universal'))

    print(f"Total sentences in dataset: {len(tagged_sents)}")
    print(f"\nFirst example sentence:\n{tagged_sents[0]}")

    universal_tags = sorted(set(tag for sent in tagged_sents for word, tag in sent))
    print(f"\nUniversal Tags: {universal_tags}")

    train_sents, test_sents = train_test_split(tagged_sents, test_size=0.2, random_state=42)

    print(f"\nWorking with Subset -> Train sentences: {len(train_sents)}, Test sentences: {len(test_sents)}")

    X_train = []
    y_train = []
    for sent in train_sents:
        words = []
        tags = []
        for w, t in sent:
            words.append(w)
            tags.append(t)
        for j in range(len(words)):
            feats = extract_features(words, j)
            X_train.append(feats)
            y_train.append(tags[j])

    X_test = []
    y_test = []
    for sent in test_sents:
        words = []
        tags = []
        for w, t in sent:
            words.append(w)
            tags.append(t)
        for j in range(len(words)):
            feats = extract_features(words, j)
            X_test.append(feats)
            y_test.append(tags[j])

    model = Pipeline([
        ('vectorizer', DictVectorizer(sparse=True)),
        ('classifier', LogisticRegression(max_iter=1000, solver='liblinear'))
    ])

    model.fit(X_train, y_train)

    print("\n=== LOGISTIC REGRESSION EVALUATION ===")
    y_pred = []
    last_5_samples = []

    print("Starting decoding for test sentences...")

    for i, sent in enumerate(test_sents):
        words = [w for w, t in sent]
        true_tags = [t for w, t in sent]

        sent_features = [extract_features(words, i) for i in range(len(words))]
        pred_tags = model.predict(sent_features)
        pred_tags_list = [str(t) for t in pred_tags]

        y_pred.extend(pred_tags_list)

        if i >= len(test_sents) - 5:
            last_5_samples.append((words, true_tags, pred_tags_list))

        if (i + 1) % 2500 == 0:
            print(f"Processed {i+1}/{len(test_sents)} sentences...")

    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n=== SAMPLE PREDICTIONS (LAST 5 SENTENCES) ===")
    for words, true, pred in last_5_samples:
        print("-" * 50)
        print(f"SENTENCE  : {' '.join(words)}")
        print(f"ORIGINAL  : {true}")
        print(f"PREDICTED : {pred}")

if __name__ == "__main__":
    main()
