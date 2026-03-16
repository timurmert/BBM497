# Timur Mert USTA - PA1

import sys
from collections import Counter

def train_BPE(file_name, init_vocab, max_merge_count=10, topK=1):

    # read the file
    f = open(file_name, 'r', encoding='utf-8')
    content = f.read()
    f.close()

    # split by whitespace
    words = content.split()

    # add _ to beginning and end of each word
    corpus = []
    for word in words:
        token = ['_'] + list(word) + ['_']
        corpus.append(token)

    merges = []
    vocabulary = list(init_vocab)

    # repeat merge operation max_merge_count times
    for repeat in range(max_merge_count):

        # count frequencies of adjacent pairs
        pair_counts = Counter()
        for word in corpus:
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += 1

        if not pair_counts: # if no pair_counts return.
            break

        # candidate list for token1, token2, frequency
        candidates = []
        for pair, count in pair_counts.items():
            candidates.append((pair[0], pair[1], count))

        def sort_key(x):
            merged = x[0] + x[1]
            return (-x[2], len(merged), merged)

        candidates.sort(key=sort_key)

        # take first topK candidates
        top_candidates = candidates[:topK]

        # which underscroe rule selected
        selected = None

        # first candidate whose merged string starts with _
        for c in top_candidates:
            merged = c[0] + c[1]
            if merged.startswith('_'):
                selected = c
                break

        # if not found first candidate whose merged string ends with _
        if selected is None:
            for c in top_candidates:
                merged = c[0] + c[1]
                if merged.endswith('_'):
                    selected = c
                    break

        # if none found, pick the first candidate
        if selected is None:
            selected = top_candidates[0]

        # add selected pair to merges list
        merges.append(selected)
        merged_token = selected[0] + selected[1]

        # add new token to vocabulary
        vocabulary.append(merged_token)

        # apply the merge to all words in corpus
        for i in range(len(corpus)):
            new_word = []
            j = 0
            while j < len(corpus[i]):
                # if adjacent tokens match the selected pair, merge them
                if j < len(corpus[i]) - 1 and corpus[i][j] == selected[0] and corpus[i][j + 1] == selected[1]:
                    new_word.append(merged_token)
                    j += 2
                else:
                    new_word.append(corpus[i][j])
                    j += 1
            corpus[i] = new_word

    return merges, vocabulary


def test_BPE(file_name, merges, vocabulary):

    # read the file
    f = open(file_name, 'r', encoding='utf-8')
    content = f.read()
    f.close()

    # split by whitespace
    words = content.split()

    # add _ to beginning and end of each word split into chars
    corpus = []
    for word in words:
        token = ['_'] + list(word) + ['_']
        corpus.append(token)

    # apply each merge rule in order
    for merge in merges:
        token1 = merge[0]
        token2 = merge[1]
        merged_token = token1 + token2

        # check merge rule for each word
        for i in range(len(corpus)):
            new_word = []
            j = 0
            while j < len(corpus[i]):
                # if match found merge them
                if j < len(corpus[i]) - 1 and corpus[i][j] == token1 and corpus[i][j + 1] == token2:
                    new_word.append(merged_token)
                    j += 2
                else:
                    new_word.append(corpus[i][j])
                    j += 1
            corpus[i] = new_word

    # flatten nested list
    tokenized_corpus = []
    for word in corpus:
        for token in word:
            tokenized_corpus.append(token)

    # find index of each token in vocabulary
    input_ids = []
    for token in tokenized_corpus:
        input_ids.append(vocabulary.index(token))

    return tokenized_corpus, input_ids


def print_truncated(lst, file=sys.stdout):
    if len(lst) > 100:
        print(f"{lst[:50]} ... {lst[-50:]}", file=file)
    else:
        print(lst, file=file)


if __name__ == "__main__":
    init_vocab = list(
        "abcçdefgğhıijklmnoöprsştuüvyzqwx"
        "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWX"
        "0123456789"
        ".,;:!?\"'()[]{}<>-_=/+*%<>^~@#$&|\\`"
    )
    init_vocab.sort()

    # ------------------------------------------------------------
    # You can use the following configurations to test your code:
    # ------------------------------------------------------------

    configs = [
        ("train.txt", "test.txt", 10, 1, "myoutput.txt"),
        ("train.txt", "test.txt", 20, 3, "myoutput.txt"),
        ("train1.txt", "test1.txt", 250, 1, "myoutput1.txt"),
        ("train1.txt", "test1.txt", 250, 5, "myoutput1.txt"),
        ("train1.txt", "test1.txt", 250, 10, "myoutput1.txt"),
        ("train2.txt", "test2.txt", 250, 1, "myoutput2.txt"),
        ("train2.txt", "test2.txt", 250, 5, "myoutput2.txt"),
        ("train2.txt", "test2.txt", 250, 10, "myoutput2.txt"),
    ]

    # Initialize or clear the output files
    for filename in ["myoutput.txt", "myoutput1.txt", "myoutput2.txt"]:
        with open(filename, "w", encoding="utf-8", newline="\n") as f:
            pass

    for i, (train_file, test_file, max_merges, top_k, out_file) in enumerate(
        configs, 1
    ):
        print(f"Running Configuration {i}: {train_file} -> {out_file}...")

        merges, vocab = train_BPE(
            train_file, init_vocab, max_merge_count=max_merges, topK=top_k
        )
        tokenized, ids = test_BPE(test_file, merges, vocab)

        # Write to consolidated file in append mode with cross-platform compatibility
        with open(out_file, "a", encoding="utf-8", newline="\n") as f:
            f.write(
                f"----- Configuration {i}: Testing with {train_file} and {test_file}: maxMerge={max_merges}, topK={top_k} -----\n"
            )
            print("Merge List:", file=f)
            print_truncated(merges, file=f)
            print("\nUpdated Vocabulary:", file=f)
            print_truncated(vocab, file=f)
            print("\nTokenized Corpus:", file=f)
            print_truncated(tokenized, file=f)
            print("\nInput IDs:", file=f)
            print_truncated(ids, file=f)
            print(f"\nToken Count: {len(tokenized)}", file=f)
            f.write("\n" + "=" * 50 + "\n\n")
