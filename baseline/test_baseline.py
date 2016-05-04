#!python3
"""
Author: Matthew Garber
Class: Natural Language Annotation for Machine Learning
Term: Spring 2016
Project: SoccEval

This program either trains a classifier or loads an existing one and tests it
on a portion of the documents from the file ratings_corpus.json, printing
information regarding the classifier's precision and recall for each class.

To run:
    python test_baseline.py <command> <classifier>

<command> should be:
    - 'new' to create the classifier from scratch.
    - 'load' to load an existing classifier.
<classifier> should be
    - 'maxent' to create or load a MaxEnt classifier.
    - 'decision_tree' to create or load a decision tree classifier.
"""

import json,  nltk, pickle, re, sys
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.metrics import scores
from nltk.tokenize import word_tokenize


STOPWORDS = stopwords.words('english')
STOPWORDS.extend([',', '.'])

ALL_RATINGS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

def unigram_boolean(document):
    """Coverts a document into a dict of unigram Boolean features.

    Args:
        document: some document's text (as a string)
    Returns:
        A dict mapping each non-stopword token to True.
    """
    tokens = [token.lower() for token in word_tokenize(document)
              if token.lower() not in STOPWORDS]
    return {token: True for token in tokens}

def unigram_freqs(document):
    """Coverts a document into a unigram frequency distribution that can be
    used as features for a classifier.

    Args:
        document: some document's text (as a string)
    Returns:
        A FreqDist of non-stopword tokens in the document.
    """
    tokens = [token.lower() for token in word_tokenize(document)
              if token.lower() not in STOPWORDS]
    return FreqDist(tokens)

def extract_features(document, feature_functions):
    """Extracts features from the document using the given feature functions.

    Args:
        document: some document's text (as a string)
        feature_functions: a list of feature functions
    Returns:
        A dict of features for the given document.
    """
    features = {}
    for feature_function in feature_functions:
        features.update(feature_function(document))  
    return features

def split_data(feature_representation, test_portion=0.1):
    """Separates the given document featuresets into a test set and a training
    set. At least one document for each label will be included in the test set.

    Args:
        feature_representation: a list of document feature sets.
        test_portion: the portion of documents to place in the test set.
    Returns:
        A tuple of the training set and test set.
    """
    if test_portion >= 1:
        raise Exception('test_portion must be less than 1!')
    cutoff = int(test_portion * len(feature_representation))
    #train_set = feature_representation[cutoff:]
    #test_set = feature_representation[:cutoff]
    train_set = []
    test_set = []
    for label in ALL_RATINGS:
        rating_set = [pair for pair in feature_representation
                      if pair[1] == label]
        cutoff = int(test_portion * len(rating_set))
        if cutoff == 0:
            cutoff += 1
        train_set.extend(rating_set[cutoff:])
        test_set.extend(rating_set[:cutoff])
    return train_set, test_set

def initialize_sets(labels):
    """Creates a dict mapping each item in labels to an empty set.
    """
    return {label : set() for label in labels}

def main(command, classifier_type):
    feature_functions = [unigram_freqs]

    corpus_file = open('ratings_corpus.json')
    corpus = json.load(corpus_file)
    corpus_file.close()

    feature_representation = [(extract_features(document, feature_functions), label)
                              for document, label in corpus]

    train_set, test_set = split_data(feature_representation)

    classifier = ''
    if command == 'new':
        if classifier_type == 'decision_tree':
            classifier = nltk.classify.DecisionTreeClassifier.train(train_set)
        elif classifier_type == 'maxent':
            classifier = nltk.classify.maxent.MaxentClassifier.train(train_set)
    elif command == 'load':
        if classifier_type == 'decision_tree':
            classifier_file = open('decisiontree_classifier.pickle', 'rb')
            classifier = pickle.load(classifier_file)
            classifier_file.close()
        elif classifier_type == 'maxent':
            classifier_file = open('maxent_classifier.pickle', 'rb')
            classifier = pickle.load(classifier_file)
            classifier_file.close()

    predictions = []
    golds = []

    for test_doc, rating in test_set:
        predictions.append(classifier.classify(test_doc))
        golds.append(rating)

    pred_sets = initialize_sets(ALL_RATINGS)
    gold_sets = initialize_sets(ALL_RATINGS)

    for doc_id, rating in enumerate(predictions):
        pred_sets[rating].add(doc_id)
    for doc_id, rating in enumerate(golds):
        gold_sets[rating].add(doc_id)

    for label in ALL_RATINGS:
        r = scores.recall(gold_sets[label], pred_sets[label])
        p = scores.precision(gold_sets[label], pred_sets[label])
        f = scores.f_measure(gold_sets[label], pred_sets[label])
        
        if not (r==None or p==None or f==None):
            f = float(f)
            print('<{}> P: {:.2}, R: {:.2}, F: {:.2}'.format(label, p, r, f))

if __name__ == '__main__':
    command = sys.argv[1]
    classifier_type = sys.argv[2]
    if command not in ['new', 'load'] or classifier_type not in ['decision_tree', 'maxent']:
        raise Exception('Invalid command-line argument')
    main(command, classifier_type)
