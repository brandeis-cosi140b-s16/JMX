#!python3
"""
Author: Matthew Garber
Class: Natural Language Annotation for Machine Learning
Term: Spring 2016
Project: SoccEval

This program either trains a classifier or loads an existing one and tests it
via k-fold cross validation on a portion of the documents from the file
ratings_corpus2.json, printing information regarding the classifier's averaged
precision and recall for each class.

NOTE: Loading previously trained classifiers doesn't not currently work correctly.

To run:
    python test_baseline.py <command> <classifier>

<command> should be:
    - 'new' to create the classifier from scratch.
    - 'load' to load an existing classifier.
<classifier> should be
    - 'maxent' to create or load a MaxEnt classifier.
    - 'decision_tree' to create or load a decision tree classifier.
    - 'lr' to create or load a linear regression classifier.
    - 'svm' to create or load a support vector machine classifier.
"""

import json, nltk, pickle, re, sys
from itertools import combinations
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.metrics import scores
from nltk.tokenize import word_tokenize
from sklearn import ensemble, linear_model, svm, tree
from sklearn.metrics import precision_recall_fscore_support

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

def select_folds(input_list, k=10):
    """Separates a list into folds for cross-validation.

    Args:
        input_list: a lsit to separate into k folds.
        k: the number of folds to create.
    Returns:
        A list of k lists, each comprising a fold.
    """
    fold_size = len(input_list) // k
    folds = []
    for i in range(0, len(input_list), fold_size):
        fold = input_list[i:i+fold_size]
        if len(fold) < fold_size:
            for x in range(len(fold)):
                folds[x].append(fold[x])
        else:
            folds.append(fold)
    return folds

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

def initialize_sets(labels):
    """Creates a dict mapping each item in labels to an empty set.
    """
    return {label : set() for label in labels}

def update_scores(avg_scores, fold_scores):
    """Updates the overall scores by adding the scores the the last iteration
    of the k-fold cross-validation.

    Args:
        avg_scores: the average cross validation scores
        fold_scores: the scores from the a cross validation iteration
    """
    avg_scores['micro']['p'] += fold_scores['micro'][0]
    avg_scores['micro']['r'] += fold_scores['micro'][1]
    avg_scores['micro']['f'] += fold_scores['micro'][2]
    avg_scores['macro']['p'] += fold_scores['macro'][0]
    avg_scores['macro']['r'] += fold_scores['macro'][1]
    avg_scores['macro']['f'] += fold_scores['macro'][2]
    for rating in ALL_RATINGS:
        avg_scores['by_label'][rating]['p'] += fold_scores['by_label'][rating]['p']
        avg_scores['by_label'][rating]['r'] += fold_scores['by_label'][rating]['r']
        avg_scores['by_label'][rating]['f'] += fold_scores['by_label'][rating]['f']
        
def average_scores(avg_scores, k=10):
    """Averages the overall scores from the k-fold cross validation.

    Args:
        avg_scores: the raw scores.
    """
    avg_scores['micro']['p'] /= k
    avg_scores['micro']['r'] /= k
    avg_scores['micro']['f'] /= k
    avg_scores['macro']['p'] /= k
    avg_scores['macro']['r'] /= k
    avg_scores['macro']['f'] /= k
    for rating in ALL_RATINGS:
        avg_scores['by_label'][rating]['p'] /= k
        avg_scores['by_label'][rating]['r'] /= k
        avg_scores['by_label'][rating]['f'] /= k

def test_iteration(i, train_set, test_dict, feature_sets_by_match,
                   classifier_type='decision_tree', command='new'):
    """Performs one iteration of the k-fold cross validation, returing a dict
    containing overall micro and macro score averages, in addition to scores for
    each label.

    For efficient retesting, this function saves pickled versions of classifier
    objects if they are trained from scratch, as they take some time to train.

    Args:
        i: the iteration of the k-fold cross validation.
        train_set: a list containing feature, rating pairs
        test_dict: a dicitonary containing feature and rating information for
            the test set.
        feature_sets_by_match: feature respresentations of documents organized
            by match.
        classifier: the type of classifier to use.
        command: 'new' to train and save a new classifier, 'load' to use a
            previously trained classifier.
    Returns:
        A dict containing overall micro and macro score averages, in addition
        to scores for each label.
    """
    #classifier = nltk.classify.scikitlearn.SklearnClassifier(svm.SVR()).train(train_set)
    classifier = ''
    if command == 'new':
        if classifier_type == 'decision_tree':
            #classifier = nltk.classify.DecisionTreeClassifier.train(train_set)
            classifier = nltk.classify.scikitlearn.SklearnClassifier(tree.DecisionTreeClassifier()).train(train_set)
        elif classifier_type == 'maxent':
            #classifier = nltk.classify.maxent.MaxentClassifier.train(train_set)
            classifier = nltk.classify.scikitlearn.SklearnClassifier(linear_model.LogisticRegression()).train(train_set)
        elif classifier_type == 'svm':
            classifier = nltk.classify.scikitlearn.SklearnClassifier(svm.SVR()).train(train_set)
        classifier_file = open('_'.join([classifier_type, str(i)]), 'wb')
        pickle.dump(classifier, classifier_file)
        classifier_file.close()
    elif command == 'load':
        classifier_file = open('_'.join([classifier_type, str(i)]), 'rb')
        classifier = pickle.load(classifier_file)
        classifier_file.close()
    
    pred_sets = initialize_sets(ALL_RATINGS)
    gold_sets = initialize_sets(ALL_RATINGS)
    pred_list = []
    gold_list = []

    # Classify predictions and add them to relevant dicts and lists.
    for match in test_dict:
        for doc_id in test_dict[match]:
            test_doc = test_dict[match][doc_id]['features']
            pred = classifier.classify(test_doc)
            gold = test_dict[match][doc_id]['gold']
            test_dict[match][doc_id]['pred'] = pred

            gold_list.append(str(gold))
            pred_list.append(str(pred))
            gold_sets[gold].add(doc_id)
            pred_sets[pred].add(doc_id)

    # Calculate pairwise ranking accuracy
    correct= 0
    total = 0
    for match in test_dict:
        for pl1, pl2 in combinations(test_dict[match].keys(), 2):
            p1 = test_dict[match][pl1]
            p2 = test_dict[match][pl2]
            if p1['gold'] > p2['gold'] and p1['pred'] > p2['pred']:
                correct += 1
            elif p1['gold'] < p2['gold'] and p1['pred'] < p2['pred']:
                correct += 1
            elif p1['gold'] == p2['gold'] and p1['pred'] == p2['pred']:
                correct += 1
            total += 1

    print('Pairwise ranking accuracy: ' + str(correct/total))
    
    fold_scores = {'micro': '',
                   'macro': '',
                   'by_label': {rating: {'p': 0, 'r': 0, 'f': 0}
                                for rating in ALL_RATINGS}
                   }
    prf_micro = precision_recall_fscore_support(gold_list, pred_list, average='micro')
    print(prf_micro)
    fold_scores['micro'] = prf_micro

    prf_macro = precision_recall_fscore_support(gold_list, pred_list, average='macro')
    print(prf_macro)
    fold_scores['macro'] = prf_macro

    for label in ALL_RATINGS:
        r = scores.recall(gold_sets[label], pred_sets[label])
        p = scores.precision(gold_sets[label], pred_sets[label])
        f = scores.f_measure(gold_sets[label], pred_sets[label])
        
        if r == None:
            r = 0.0
        if p == None:
            p = 0.0
        if f == None:
            f = 0.0
            
        fold_scores['by_label'][label]['p'] = p
        fold_scores['by_label'][label]['r'] = r
        fold_scores['by_label'][label]['f'] = f
        f = float(f)
        print('<{}> P: {:.3}, R: {:.3}, F: {:.3}'.format(label, p, r, f))

    return fold_scores

def main(command, classifier_type):
    
    feature_functions = [unigram_boolean]
    
    corpus_file = open('ratings_corpus2.json')
    corpus = json.load(corpus_file)
    corpus_file.close()

    matches = set()
    for doc_id in corpus:
        matches.add(doc_id.split('_')[0])
    
    feature_sets_by_match = {}
    for doc_id, entry in corpus.items():
        text = entry[0]
        rating = entry[1]
        features = extract_features(text, feature_functions)

        prefix = doc_id.split('_')[0]
        if prefix not in feature_sets_by_match:
            feature_sets_by_match[prefix] = {doc_id: (features, rating)}
        else:
            feature_sets_by_match[prefix][doc_id] = (features, rating)

    avg_scores = {'micro': {'p': 0, 'r': 0, 'f': 0},
                  'macro': {'p': 0, 'r': 0, 'f': 0},
                  'by_label': {rating: {'p': 0, 'r': 0, 'f': 0}
                               for rating in ALL_RATINGS}
                  }

    folds = select_folds(sorted(matches), 15)

    for i in range(len(folds)):
        train_folds = folds[:i]
        train_folds.extend(folds[i+1:])
        train_set = []
        for fold in train_folds:
            for match in fold:
                for doc_id in feature_sets_by_match[match]:
                    train_set.append(feature_sets_by_match[match][doc_id])
        test_dict = {}
        for match in folds[i]:
            test_dict[match] = {}
            for doc_id in feature_sets_by_match[match]:
                test_dict[match][doc_id] = {'features': feature_sets_by_match[match][doc_id][0]}
                test_dict[match][doc_id]['gold'] = feature_sets_by_match[match][doc_id][1]

        print('iteration {}:'.format(i))
        fold_scores = test_iteration(i, train_set, test_dict, feature_sets_by_match,
                                     classifier_type=classifier_type, command=command)
        update_scores(avg_scores, fold_scores)
        
    average_scores(avg_scores, k=15)
    micro = [avg_scores['micro'][label] for label in ['p', 'r', 'f']]
    macro = [avg_scores['macro'][label] for label in ['p', 'r', 'f']]
    print('MICRO P: {:.3}, R: {:.3}, F: {:.3}'.format(micro[0], micro[1], micro[2]))
    print('MACRO P: {:.3}, R: {:.3}, F: {:.3}'.format(macro[0], macro[1], macro[2]))
    for label in ALL_RATINGS:
        p = avg_scores['by_label'][label]['p']
        r = avg_scores['by_label'][label]['r']
        f = avg_scores['by_label'][label]['f']
        print('<{}> P: {:.3}, R: {:.3}, F: {:.3}'.format(label, p, r, f))
    

if __name__ == '__main__':
    command = sys.argv[1]
    classifier_type = sys.argv[2]
    if command not in ['new', 'load'] or classifier_type not in ['decision_tree', 'lr', 'maxent', 'svm']:
        raise Exception('Invalid command-line argument')
    main(command, classifier_type)
