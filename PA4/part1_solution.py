import gensim.downloader
import numpy as np


model = gensim.downloader.load("word2vec-google-news-300")
EMBEDDING_DIM = model.vector_size


def replace_with_similar(sentence, indices):
    tokens = sentence.split()

    similar_words_dict = {}
    for idx in indices:
        target_word = tokens[idx]
        candidates = model.most_similar(target_word, topn=5)

        candidate_list = []
        for pair in candidates:
            similar_word = pair[0]
            score = pair[1]
            candidate_list.append((similar_word, score))

        similar_words_dict[target_word] = candidate_list

    new_tokens = tokens[:]
    for idx in indices:
        original_word = tokens[idx]
        best_replacement = similar_words_dict[original_word][0][0]
        new_tokens[idx] = best_replacement

    new_sentence = " ".join(new_tokens)
    return new_sentence, similar_words_dict


def sentence_vector(sentence):
    tokens = sentence.split()

    vector_dict = {}
    for word in tokens:
        if word in model:
            vector_dict[word] = model[word]
        else:
            vector_dict[word] = np.zeros(EMBEDDING_DIM)

    total = np.zeros(EMBEDDING_DIM)
    for word in vector_dict:
        total = total + vector_dict[word]

    sentence_vec = total / len(vector_dict)
    return vector_dict, sentence_vec


def most_similar_sentences(file_path, query):
    file_handle = open(file_path, "r", encoding="utf-8")
    raw_lines = file_handle.readlines()
    file_handle.close()

    sentences = []
    for line in raw_lines:
        cleaned_line = line.strip()
        if cleaned_line != "":
            sentences.append(cleaned_line)

    _, query_vec = sentence_vector(query)

    results = []
    for sentence in sentences:
        _, sent_vec = sentence_vector(sentence)

        dot_product = np.dot(query_vec, sent_vec)
        query_norm = np.linalg.norm(query_vec)
        sent_norm = np.linalg.norm(sent_vec)

        if query_norm == 0 or sent_norm == 0:
            cosine_score = 0.0
        else:
            cosine_score = dot_product / (query_norm * sent_norm)

        results.append((sentence, float(cosine_score)))

    def take_score(item):
        return item[1]

    results.sort(key=take_score, reverse=True)
    return results


def analyze_dimension_contributions(word1, word2, top_k=10):
    u = model[word1]
    v = model[word2]

    contributions = []
    for i in range(len(u)):
        product = u[i] * v[i]
        contributions.append((i, float(product), float(u[i]), float(v[i])))

    def by_abs(item):
        return abs(item[1])

    contributions.sort(key=by_abs, reverse=True)

    top_results = []
    for i in range(top_k):
        top_results.append(contributions[i])

    return top_results
