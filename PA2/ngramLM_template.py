# -*- coding: utf-8 -*-

import math
import random
import re
import codecs

"""
Important:
You are free to add new variables, helper functions, or additional methods if
needed for your implementation. However, the required methods and instance
variables provided in this template must remain unchanged and must be properly
implemented.
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
        # TO DO: data structures for ngrams

    # INSTANCE METHODS
    def trainFromFile(self, fn):
        # TO DO
        return

    def vocab(self):
        # TO DO
        return

    def bigrams(self):
        # TO DO
        return

    def trigrams(self):
        # TO DO
        return

    def unigramCount(self, word):
        # TO DO
        return

    def bigramCount(self, bigram):
        # TO DO
        return

    def trigramCount(self, trigram):
        # TO DO
        return

    def unigramProb(self, word):
        # TO DO
        # returns unsmoothed unigram probability value
        return

    def bigramProb(self, bigram):
        # TO DO
        # returns unsmoothed bigram probability value
        return

    def trigramProb(self, trigram):
        # TO DO
        # returns unsmoothed trigram probability value
        return

    def unigramProb_SmoothingUNK(self, word):
        # TO DO
        # returns smoothed unigram probability value
        return

    def bigramProb_SmoothingUNK(self, bigram):
        # TO DO
        # returns smoothed bigram probability value
        return

    def trigramProb_SmoothingUNK(self, trigram):
        # TO DO
        # returns smoothed trigram probability value
        return

    def sentenceProb(self, sent, model="bigram"):
        # TO DO
        # sent is a list of tokens
        # returns the probability of sent
        return

    def perplexity(self, testFile, model="bigram"):
        # TO DO
        # returns perplexity value
        return

    def generateSentence(self, sent=["<s>"], maxFollowWords=3, maxWordsInSent=20):
        # TO DO
        # returns the generated sentence (a list of tokens)
        return