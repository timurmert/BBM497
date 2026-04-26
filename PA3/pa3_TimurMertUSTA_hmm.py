import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from collections import defaultdict, Counter

nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

class HMMTagger:
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.emissions = defaultdict(Counter)
        self.tags = set()
        self.vocab = set()
        self.tag_counts = Counter()

    def train(self, tagged_sents):
        for sent in tagged_sents:
            prev_tag = "<START>"
            for word, tag in sent:
                self.transitions[prev_tag][tag] += 1
                self.emissions[tag][word] += 1
                self.tag_counts[tag] += 1
                self.tags.add(tag)
                self.vocab.add(word)
                prev_tag = tag

    def get_emission_prob(self, word, tag):
        c_t = self.tag_counts[tag]
        v_size = len(self.vocab)

        if word not in self.vocab:
            prob = 1.0 / (c_t + v_size)
        else:
            c_tw = self.emissions[tag][word]
            prob = (c_tw + 1e-10) / c_t

        return np.log(prob)

    def get_transition_prob(self, prev_tag, curr_tag):

        c_prev_curr = self.transitions[prev_tag][curr_tag]

        total_from_prev = 0
        for t in self.transitions[prev_tag]:
            total_from_prev += self.transitions[prev_tag][t]

        t_size = len(self.tags)

        prob = (c_prev_curr + 1) / (total_from_prev + t_size)

        return np.log(prob)

    def viterbi(self, sentence):
        tags_list = sorted(self.tags)
        m = len(tags_list)
        n = len(sentence)

        viterbi = np.full((m, n), -np.inf)
        backpointer = np.zeros((m, n), dtype=int)

        for i in range(m):
            t_i = tags_list[i]
            trans_p = self.get_transition_prob("<START>", t_i)
            emit_p = self.get_emission_prob(sentence[0], t_i)
            viterbi[i, 0] = trans_p + emit_p

        for j in range(1, n):
            for i in range(m):
                t_i = tags_list[i]
                emit_p = self.get_emission_prob(sentence[j], t_i)

                scores = np.full(m, -np.inf)
                for k in range(m):
                    t_k = tags_list[k]
                    trans_p = self.get_transition_prob(t_k, t_i)
                    scores[k] = viterbi[k, j - 1] + trans_p + emit_p

                best_k = np.argmax(scores)
                viterbi[i, j] = scores[best_k]
                backpointer[i, j] = best_k

        best_last = np.argmax(viterbi[:, n - 1])

        best_path_indices = [0] * n
        best_path_indices[n - 1] = best_last
        for j in range(n - 1, 0, -1):
            best_path_indices[j - 1] = backpointer[best_path_indices[j], j]

        best_path = []
        for idx in best_path_indices:
            best_path.append(tags_list[idx])

        return best_path

def main():

    tagged_sents = list(brown.tagged_sents(tagset='universal'))

    print(f"Total sentences in dataset: {len(tagged_sents)}")
    print(f"\nFirst example sentence:\n{tagged_sents[0]}")

    universal_tags = sorted(set(tag for sent in tagged_sents for word, tag in sent))
    print(f"\nUniversal Tags: {universal_tags}")

    train_sents, test_sents = train_test_split(tagged_sents, test_size=0.2, random_state=42)

    print(f"\nWorking with Subset -> Train sentences: {len(train_sents)}, Test sentences: {len(test_sents)}")

    tagger = HMMTagger()
    tagger.train(train_sents)

    transitions_summary = {k: dict(sorted(v.items())[:5]) for k, v in sorted(tagger.transitions.items())[:5]}
    emissions_summary = {k: dict(sorted(v.items())[:5]) for k, v in sorted(tagger.emissions.items())[:5]}

    print(f"\nTagger Transitions (first 5 sorted): {transitions_summary}")
    print(f"\nTagger Emissions (first 5 sorted): {emissions_summary}")
    print(f"\nTagger Tags (sorted): {sorted(list(tagger.tags))}")
    print(f"\nVocabulary size: {len(tagger.vocab)}")
    print(f"\nTagger Vocab (first 5 sorted): {sorted(list(tagger.vocab))[:5]}")
    print(f"\nTagger Tag Counts (sorted items): {dict(sorted(tagger.tag_counts.items()))}")

    print("\n=== VITERBI DECODING & EVALUATION ===")
    y_true = []
    y_pred = []
    last_5_samples = []

    print("Starting decoding for test sentences...")
    for i, sent in enumerate(test_sents):
        words = [w for w, t in sent]
        true_tags = [t for w, t in sent]

        pred_tags = tagger.viterbi(words)

        y_true.extend(true_tags)
        y_pred.extend(pred_tags)

        if i >= len(test_sents) - 5:
            last_5_samples.append((words, true_tags, pred_tags))

        if (i + 1) % 2500 == 0:
            print(f"Processed {i+1}/{len(test_sents)} sentences...")

    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("=== SAMPLE PREDICTIONS (LAST 5 SENTENCES) ===")
    for words, true, pred in last_5_samples:
        print("-" * 50)
        print(f"SENTENCE  : {' '.join(words)}")
        print(f"ORIGINAL  : {true}")
        print(f"PREDICTED : {pred}")

if __name__ == "__main__":
    main()
