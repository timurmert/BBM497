# -*- coding: utf-8 -*-

import math
import random
import re
import codecs
import tokenize

"""
Important:
You are free to add new variables, helper functions, or additional methods if
needed for your implementation. However, the required methods and instance
variables provided in this template must remain unchanged and must be properly
implemented.
"""

pattern = r"""(?x)
(?:[A-ZÇĞIİÖŞÜ]\.)+
| \d+(?:\.\d*)?(?:\’\w+)?
| \w+(?:-\w+)*(?:\’\w+)?
| \.\.\.
| [][,;.?():_!#^+$%&><|/{()=}\"\’\\\"\‘-]
"""

# ngramLM CLASS
class ngramLM:
    """Ngram Language Model Class"""
    
    # Create Empty ngramLM
    def __init__(self):
        self.numOfTokens = 0
        self.sizeOfVocab = 0
        self.numOfSentences = 0
        self.sentences = []
        self.tokenized_lines = []
        # TO DO: data structures for ngrams
        self.unigramCounts = {}
        self.bigramCounts = {}
        self.trigramCounts = {}

    # INSTANCE METHODS
    def trainFromFile(self, fn):

        with open(fn, encoding="utf-8") as f:

            # lowercase op and tokenization
            for sentence in f.readlines():
                sentence = sentence.replace("I", "ı").replace("İ", "i").lower()
                tokenized = re.findall(pattern, sentence)
                if len(tokenized) != 0:
                    self.tokenized_lines.append(tokenized)

            # seperating sentences and append <s> and </s> for the structure
            for line_tokens in self.tokenized_lines:
                current_sentence = ["<s>"]

                for token in line_tokens:
                    current_sentence.append(token)

                    if token in [".","!","?"]:
                        current_sentence.append("</s>")
                        self.sentences.append(current_sentence)
                        self.numOfTokens += len(current_sentence)
                        current_sentence = ["<s>"]

                if current_sentence != ["<s>"]:
                    current_sentence.append("</s>")
                    self.sentences.append(current_sentence)
                    self.numOfTokens += len(current_sentence)

            self.numOfSentences = len(self.sentences)

            # unigram counts
            for sentence in self.sentences:
                for token in sentence:
                    if not (token in self.unigramCounts):
                        self.unigramCounts[token] = 1
                    else:
                        self.unigramCounts[token] += 1

            # bigram counts
            for sentence in self.sentences:
                for i in range(len(sentence) - 1):
                    bigram = (sentence[i], sentence[i+1])

                    if bigram not in self.bigramCounts:
                        self.bigramCounts[bigram] = 1
                    else:
                        self.bigramCounts[bigram] += 1

            # trigram counts
            for sentence in self.sentences:
                for i in range (len(sentence) - 2):
                    trigram = (sentence[i], sentence[i+1], sentence[i+2])

                    if trigram not in self.trigramCounts:
                        self.trigramCounts[trigram] = 1
                    else:
                        self.trigramCounts[trigram] += 1

            # vocabulary size
            self.sizeOfVocab = len(self.unigramCounts.keys())

        return

    def sort(self, item):
        word = item[0]
        frequency = item[1]

        return(-frequency, word)

    def vocab(self):
        
        vocab_list = list(self.unigramCounts.items())
        vocab_list.sort(key=self.sort)

        return vocab_list

    def bigrams(self):
        
        bigrams_list = list(self.bigramCounts.items())
        bigrams_list.sort(key=self.sort)

        return bigrams_list

    def trigrams(self):

        trigrams_list = list(self.trigramCounts.items())
        trigrams_list.sort(key=self.sort)
        
        return trigrams_list

    def unigramCount(self, word):
        
        if word in self.unigramCounts.keys():
            return self.unigramCounts[word]
        else:
            return 0

    def bigramCount(self, bigram):

        if bigram in self.bigramCounts.keys():
            return self.bigramCounts[bigram]
        else:
            return 0

    def trigramCount(self, trigram):
        
        if trigram in self.trigramCounts.keys():
            return self.trigramCounts[trigram]
        else:
            return 0

    def unigramProb(self, word):

        if self.numOfTokens == 0:
            return 0

        return self.unigramCount(word) / self.numOfTokens

    def bigramProb(self, bigram):
        
        denominator = self.unigramCount(bigram[0])

        if denominator == 0:
            return 0

        return self.bigramCount(bigram) / denominator

    def trigramProb(self, trigram):

        denominator = self.bigramCount(trigram[0:2])

        if denominator == 0:
            return 0
        
        return self.trigramCount(trigram) / self.bigramCount(trigram[0:2])

    def unigramProb_SmoothingUNK(self, word):
        
        return (self.unigramCount(word) + 1) / (self.numOfTokens + (self.sizeOfVocab + 1))

    def bigramProb_SmoothingUNK(self, bigram):
        
        return (self.bigramCount(bigram) + 1) / (self.unigramCount(bigram[0]) + (self.sizeOfVocab + 1))

    def trigramProb_SmoothingUNK(self, trigram):
        
        return (self.trigramCount(trigram) + 1) / (self.bigramCount(trigram[0:2]) + (self.sizeOfVocab + 1))

    def sentenceProb(self, sent, model="bigram"):
        # TO DO
        # sent is a list of tokens
        # returns the probability of sent

        prob = 1.0

        if model == "bigram":
            if len(sent) == 1:
                return self.unigramProb_SmoothingUNK(sent[0])

            for i in range(1, len(sent)):
                word_1 = sent[i-1]
                word_2 = sent[i]
                prob = prob * self.bigramProb_SmoothingUNK((word_1, word_2))

        elif model == "trigram":
            if len(sent) == 1:
                return self.unigramProb_SmoothingUNK(sent[0])

            for i in range(2, len(sent)):
                word_1 = sent[i-2]
                word_2 = sent[i-1]
                word_3 = sent[i]

                prob = prob * self.trigramProb_SmoothingUNK((word_1, word_2, word_3))

        return prob

    def perplexity(self, testFile, model="bigram"):
        # TO DO
        # returns perplexity value

        tokenized_lines = []
        sentences = []
        numOfTokens = 0
        log_prob_sum = 0.0

        with open(testFile, encoding="utf-8") as f:

            # lowercase op and tokenization
            for sentence in f.readlines():
                sentence = sentence.replace("I", "ı").replace("İ", "i").lower()
                tokenized = re.findall(pattern, sentence)
                if len(tokenized) != 0:
                    tokenized_lines.append(tokenized)

            # seperating sentences and append <s> and </s> for the structure
            for line_tokens in tokenized_lines:
                current_sentence = ["<s>"]

                for token in line_tokens:
                    current_sentence.append(token)

                    if token in [".","!","?"]:
                        current_sentence.append("</s>")
                        sentences.append(current_sentence)
                        numOfTokens += len(current_sentence)
                        current_sentence = ["<s>"]

                if current_sentence != ["<s>"]:
                    current_sentence.append("</s>")
                    sentences.append(current_sentence)
                    numOfTokens += len(current_sentence)

        if model == "bigram":
            for sentence in sentences:
                for i in range(1, len(sentence)):
                    word_1 = sentence[i-1]
                    word_2 = sentence[i]
                    log_prob_sum += math.log(self.bigramProb_SmoothingUNK((word_1, word_2)))

        elif model == "trigram":
            for sentence in sentences:
                for i in range(2, len(sentence)):
                    word_1 = sentence[i-2]
                    word_2 = sentence[i-1]
                    word_3 = sentence[i]

                    log_prob_sum += math.log(self.trigramProb_SmoothingUNK((word_1, word_2, word_3)))

        return math.exp(-log_prob_sum / numOfTokens)

    def generateSentence(self, sent=["<s>"], maxFollowWords=3, maxWordsInSent=20):

        generated = sent.copy()

        if maxWordsInSent == 0:
            if generated[-1] != "</s>":
                generated.append("</s>")
            return generated

        while True:
            candidates = []
            current_word = generated[-1]
            bigram_list = self.bigrams()

            for bigram in bigram_list:
                if bigram[0][0] == current_word:
                    candidates.append((bigram[0][1], bigram[1]))

            candidates = candidates[0:maxFollowWords]

            if len(candidates) == 0:
                if generated[-1] != "</s>":
                    generated.append("</s>")
                break

            words = []
            weights = []

            for word, freq in candidates:
                words.append(word)
                weights.append(freq)

            next_word = random.choices(words, weights=weights, k=1)[0]
            generated.append(next_word)

            if next_word == "</s>":
                break

            normal_word_count = len(generated) - len(sent)

            if normal_word_count >= maxWordsInSent:
                if generated[-1] != "</s>":
                    generated.append("</s>")
                break

        return generated

lm = ngramLM()
lm.trainFromFile("tinyTestCorpus.txt")

with open("tinyTest_output.txt", "w", encoding="utf-8") as f:
    print("LM numOfTokens: ", lm.numOfTokens, file=f)
    print("LM sizeOfVocab: ", lm.sizeOfVocab, file=f)
    print("LM numOfSentences: ", lm.numOfSentences, file=f)
    print("LM Sentences: \n", lm.sentences, file=f)
    print("LM Sorted Vocabulary (Unigrams) with Frequencies: \n", lm.vocab(), file=f)
    print("LM Sorted Bigrams with Frequencies: \n", lm.bigrams(), file=f)

    print("unigramCount('a'):", lm.unigramCount('a'), file=f)
    print("unigramCount('b'):", lm.unigramCount('b'), file=f)
    print("unigramCount('g'):", lm.unigramCount('g'), file=f)

    print("unigramProb('a'):", lm.unigramProb('a'), file=f)
    print("unigramProb('b'):", lm.unigramProb('b'), file=f)
    print("unigramProb('g'):", lm.unigramProb('g'), file=f)

    print("bigramCount(('a','b')):", lm.bigramCount(('a','b')), file=f)
    print("bigramCount(('b','a')):", lm.bigramCount(('b','a')), file=f)
    print("bigramCount(('a','g')):", lm.bigramCount(('a','g')), file=f)
    print("bigramCount(('g','a')):", lm.bigramCount(('g','a')), file=f)
    print("bigramCount(('g','g')):", lm.bigramCount(('g','g')), file=f)

    print("bigramProb(('a','b')):", lm.bigramProb(('a','b')), file=f)
    print("bigramProb(('b','a')):", lm.bigramProb(('b','a')), file=f)
    print("bigramProb(('g','a')):", lm.bigramProb(('g','a')), file=f)
    print("bigramProb(('a','g')):", lm.bigramProb(('a','g')), file=f)
    print("bigramProb(('g','g')):", lm.bigramProb(('g','g')), file=f)

    print("unigramProb_SmoothingUNK('a'):", lm.unigramProb_SmoothingUNK('a'), file=f)
    print("unigramProb_SmoothingUNK('b'):", lm.unigramProb_SmoothingUNK('b'), file=f)
    print("unigramProb_SmoothingUNK('g'):", lm.unigramProb_SmoothingUNK('g'), file=f)

    print("bigramProb_SmoothingUNK(('a','b')):", lm.bigramProb_SmoothingUNK(('a','b')), file=f)
    print("bigramProb_SmoothingUNK(('b','a')):", lm.bigramProb_SmoothingUNK(('b','a')), file=f)
    print("bigramProb_SmoothingUNK(('g','a')):", lm.bigramProb_SmoothingUNK(('g','a')), file=f)
    print("bigramProb_SmoothingUNK(('a','g')):", lm.bigramProb_SmoothingUNK(('a','g')), file=f)
    print("bigramProb_SmoothingUNK(('g','g')):", lm.bigramProb_SmoothingUNK(('g','g')), file=f)

    print("sentenceProb(['<s>','a','f','d','.','</s>']):", lm.sentenceProb(['<s>','a','f','d','.','</s>']), file=f)
    print("sentenceProb(['<s>','a','c','d','.','</s>']):", lm.sentenceProb(['<s>','a','c','d','.','</s>']), file=f)
    print("sentenceProb(['<s>','a','b','c','d','.','</s>']):", lm.sentenceProb(['<s>','a','b','c','d','.','</s>']), file=f)
    print("sentenceProb(['<s>','</s>']):", lm.sentenceProb(['<s>','</s>']), file=f)
    print("sentenceProb(['<s>']):", lm.sentenceProb(['<s>']), file=f)
    print("sentenceProb(['a']):", lm.sentenceProb(['a']), file=f)

    print("generateSentence():", lm.generateSentence(), file=f)
    print("generateSentence([\"<s>\"],2,20):", lm.generateSentence(["<s>"],2,20), file=f)
    print("generateSentence([\"<s>\"],2,20):", lm.generateSentence(["<s>"],2,20), file=f)
    print("generateSentence([\"<s>\"],3,20):", lm.generateSentence(["<s>"],3,20), file=f)
    print("generateSentence([\"<s>\"],3,20):", lm.generateSentence(["<s>"],3,20), file=f)
    print("generateSentence([\"<s>\"],2,2):", lm.generateSentence(["<s>"],2,2), file=f)
    print("generateSentence([\"<s>\"],2,2):", lm.generateSentence(["<s>"],2,2), file=f)
    print("generateSentence([\"<s>\"],2,2):", lm.generateSentence(["<s>"],2,2), file=f)
    print("generateSentence([\"<s>\"],2,1):", lm.generateSentence(["<s>"],2,1), file=f)
    print("generateSentence([\"<s>\"],2,1):", lm.generateSentence(["<s>"],2,1), file=f)
    print("generateSentence([\"<s>\"],2,0):", lm.generateSentence(["<s>"],2,0), file=f)