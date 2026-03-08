import sys
from collections import Counter

def train_BPE(file_name, init_vocab, max_merge_count=10, topK=1):

    # TO DO

    return merges, vocabulary


def test_BPE(file_name, merges, vocabulary):
    
    # TO DO

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

    # configs = [
    #     ("train.txt", "test.txt", 10, 1, "myoutput.txt"),
    #     ("train.txt", "test.txt", 20, 3, "myoutput.txt"),
    #     ("train1.txt", "test1.txt", 250, 1, "myoutput1.txt"),
    #     ("train1.txt", "test1.txt", 250, 5, "myoutput1.txt"),
    #     ("train1.txt", "test1.txt", 250, 10, "myoutput1.txt"),
    #     ("train2.txt", "test2.txt", 250, 1, "myoutput2.txt"),
    #     ("train2.txt", "test2.txt", 250, 5, "myoutput2.txt"),
    #     ("train2.txt", "test2.txt", 250, 10, "myoutput2.txt"),
    # ]

    # # Initialize or clear the output files
    # for filename in ["myoutput.txt", "myoutput1.txt", "myoutput2.txt"]:
    #     with open(filename, "w", encoding="utf-8", newline="\n") as f:
    #         pass

    # for i, (train_file, test_file, max_merges, top_k, out_file) in enumerate(
    #     configs, 1
    # ):
    #     print(f"Running Configuration {i}: {train_file} -> {out_file}...")

    #     merges, vocab = train_BPE(
    #         train_file, init_vocab, max_merge_count=max_merges, topK=top_k
    #     )
    #     tokenized, ids = test_BPE(test_file, merges, vocab)

    #     # Write to consolidated file in append mode with cross-platform compatibility
    #     with open(out_file, "a", encoding="utf-8", newline="\n") as f:
    #         f.write(
    #             f"----- Configuration {i}: Testing with {train_file} and {test_file}: maxMerge={max_merges}, topK={top_k} -----\n"
    #         )
    #         print("Merge List:", file=f)
    #         print_truncated(merges, file=f)
    #         print("\nUpdated Vocabulary:", file=f)
    #         print_truncated(vocab, file=f)
    #         print("\nTokenized Corpus:", file=f)
    #         print_truncated(tokenized, file=f)
    #         print("\nInput IDs:", file=f)
    #         print_truncated(ids, file=f)
    #         print(f"\nToken Count: {len(tokenized)}", file=f)
    #         f.write("\n" + "=" * 50 + "\n\n")
