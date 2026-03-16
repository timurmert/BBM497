import sys
from collections import Counter

def train_BPE(file_name, init_vocab, max_merge_count=10, topK=1):
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.read()

    words = content.split()
    corpus = [['_'] + list(word) + ['_'] for word in words]

    merges = []
    vocabulary = list(init_vocab)

    for _ in range(max_merge_count):
        pair_counts = Counter()
        for word in corpus:
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += 1

        if not pair_counts:
            break

        candidates = [(p[0], p[1], c) for p, c in pair_counts.items()]
        candidates.sort(key=lambda x: (-x[2], len(x[0] + x[1]), x[0] + x[1]))

        top_candidates = candidates[:topK]

        selected = None
        for c in top_candidates:
            if (c[0] + c[1]).startswith('_'):
                selected = c
                break
        if selected is None:
            for c in top_candidates:
                if (c[0] + c[1]).endswith('_'):
                    selected = c
                    break
        if selected is None:
            selected = top_candidates[0]

        merges.append(selected)
        merged_token = selected[0] + selected[1]
        vocabulary.append(merged_token)

        for i, word in enumerate(corpus):
            new_word = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == selected[0] and word[j + 1] == selected[1]:
                    new_word.append(merged_token)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            corpus[i] = new_word

    return merges, vocabulary


def test_BPE(file_name, merges, vocabulary):
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.read()

    words = content.split()
    corpus = [['_'] + list(word) + ['_'] for word in words]

    for merge in merges:
        token1, token2 = merge[0], merge[1]
        merged_token = token1 + token2
        for i, word in enumerate(corpus):
            new_word = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == token1 and word[j + 1] == token2:
                    new_word.append(merged_token)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            corpus[i] = new_word

    tokenized_corpus = []
    for word in corpus:
        tokenized_corpus.extend(word)

    input_ids = [vocabulary.index(token) for token in tokenized_corpus]

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
