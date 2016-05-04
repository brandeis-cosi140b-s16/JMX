#!python3
"""
Author: Matthew Garber
Class: Natural Language Annotation for Machine Learning
Term: Spring 2016
Project: SoccEval

This program creates a JSON corpus of sentences regarding a particular player
during a given match and that player's rating for that match, which can be used
to train a classifier. The corpus is saved as ratings_corpus.json.

To run:
    python create_baseline_corpus.py

"""

import json
import os
import re
import unicodedata
from nltk import sent_tokenize
from urllib.parse import unquote_plus

coreferential_terms = ['defender', 'forward' 'goalie', 'goalkeeper',
                       'he', 'him', 'his', 'keeper', 'midfielder',
                       'player', 'striker']
pattern_string = r'\b(' + '|'.join(coreferential_terms) + r')\b' 
coref_pattern = re.compile(pattern_string, flags=re.I)

def get_match_filepath(ratings_path, source):
    """Retrieves the filepath for the match report corresponding to a given
    ratings file.

    Args:
        rating_path: the full filepath to the ratings file.
        source: the source of the match report, either 'goal' or 'guardian'
    Returns:
        a string filepath to the match report
    """
    text_match = re.search(r'/[^/]*$', ratings_path)
    ratings_filename = text_match.group(0)
    match_filename = ratings_filename[2:]
    match_path = ''
    if source == 'goal':
        match_path = r'../goal_articles/all/' + match_filename
    elif source == 'guardian':
        date_match = re.match(r'^(\d{4})(\d{2})(\d{2})', match_filename)
        date = '-'.join(date_match.groups())
        filename_with_spaces = match_filename.replace('-', ' ')
        team1, team2 = re.search(r'([a-z\- ]+)? vs ([a-z\- ]+)',
                                 filename_with_spaces).groups()
        team_dict = {
            'queens park rangers': 'qpr',
            'tottenham hotspur': 'tottenham',
            'west ham united': 'west ham',
            }
        if team1 in team_dict:
            team1 = team_dict[team1]
        if team2 in team_dict:
            team2 = team_dict[team2]
        guardian_root = r'../guardian_articles/matchreportsCLT/'
        for root, dirs, files in os.walk(guardian_root):
            if root == guardian_root:
                for f in files:
                    if f.startswith(date):
                        match1 = re.search(team1, f, re.I)
                        match2 = re.search(team2, f, re.I)
                        if match1 and match2:
                            match_path = guardian_root + f
    else:
        raise Exception("source must be 'goal' or 'guardian'")
    return match_path

def parse_ratings(ratings_file):
    """Parses the ratings file, returning a dictionary mapping a tuple of the
    player's full, first, and last names to the corresponding rating.

    Args:
        ratings_file: a file containing the rating of soccer players for a
            given match.
    Returns:
        A dicitionary mapping a player_id to a their rating and tuple of the
        player's names 
    """
    player_info = {}
    pattern = re.compile(r'^href="/en-us/people/[\w-]+/\d+/(.*?)(-(.*?))?" data-rate="(\d\.\d)"')
    
    player_id = 0
    for line in ratings_file:
        clean_line = strip_accents(unquote_plus(line))
        match = re.match(pattern, clean_line)
        first_name = match.group(1)
        if match.group(3):
            last_name = match.group(3)
        else:
            last_name = first_name
        full_name = ' '.join([first_name, last_name])
        names = tuple(wordify(word) for word in [full_name, first_name, last_name])
        rating = float(match.group(4))
        player_info[player_id] = {
            'names': names,
            'rating': rating
            }
        player_id += 1

    return player_info

def create_subdocs(match_text, player_info, source):
    """Creates a sub-document for each player mentioned in the document, and
    returns a list of tuples of sub-documents and player ratings.

    Sentences that mention multiple players will be added multiple times,
    regardless of whether they are rated the same or differently.

    Args:
        match_text: text of a match report article.
        player_info: a dict mapping plauer_id to a rating and names fields.
        source: 'goal' if the match text is from a goal.com article or
            'guardian' if it is from a Guardian article.
    Returns:
        A list of tuples pairing sentences about a player with that player's
        rating for the match.
    """
    if source not in ('goal', 'guardian'):
        raise Exception('Invalid source')
    cleaned_text = remove_header(match_text, source)
    cleaned_text = strip_accents(cleaned_text)
    sents = sent_tokenize(cleaned_text)

    subdocs_by_player = {}
    latest = [-1, -1]   # [player_id, span_end]
    has_player_name = False
    for sent in sents:
        for player_id, fields in player_info.items():
            i = 0
            match = False
            while i < 3 and not match:
                match = re.search(player_info[player_id]['names'][i], sent, re.I)
                i += 1
            if match:
                has_player_name = True
                dict_append(subdocs_by_player, player_id, sent)
                if latest[1] > match.end():
                    latest = [player_id, match.end()]
        if not has_player_name:
            match = re.search(coref_pattern, sent)
            if match and latest[1] > -1:
                dict_append(subdocs_by_player, player_id, sent)
                
    # Convert subdocs_by_player into tuples of (subdoc, player_rating)
    subdocs = []
    for player_id, sents in subdocs_by_player.items():
        text = ' '.join(subdocs_by_player[player_id])
        subdocs.append((text, player_info[player_id]['rating']))
    return subdocs

def create_ratings_corpus(ratings_filenames, source):
    """Creates a labeled corpus of sub-documents from documents corresponding
    to the given list of ratings file filenames.
    
    For each player in each document, all sentences regarding that player are
    concatenated into a sub-document and labeled with that player's rating for
    that match.

    Args:
        ratings_filenames: a list of filenames of ratings files.
        source: 'goal' to use source documents from goal.com or 'guardian' to
            use source documents from the Guardian.
    Returns:
        The corpus as a list of sub-document, rating pairs.
    """
    subdocs = []
    for filepath in ratings_filenames:
        ratings_file = open(filepath)
        player_info = parse_ratings(ratings_file)
        match_filepath = get_match_filepath(filepath, source)
        if match_filepath:
            match_text = open(match_filepath, encoding='utf8').read()
            subdocs.extend(create_subdocs(match_text, player_info, source))
    return subdocs

def strip_accents(string):
    """Removes accents from a given string.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', string)
                  if unicodedata.category(c) != 'Mn')

def wordify(word):
    """Adds regex word boundary symbols to the start and end of the given
    string.
    """
    return r'\b' + word + r'\b'

def dict_append(dict, key, value):
    """Appends a value to the list corresponding to the given key in the given
    dictionary. Creates a new list if there is no such key.
    """
    if key not in dict:
        dict[key] = [value]
    else:
        dict[key].append(value)

def remove_header(text, source):
    """Removes the header from the text of a given match report. 'source'
    designates the source of that match report.
    """
    pattern = ''
    if source == 'goal':
        pattern = re.compile(r'^.*?[AP]M\n', re.DOTALL)
    elif source == 'guardian':
        pattern = re.compile(r'^.*?\n')
    else:
        raise Exception("source must be 'goal' or 'guardian'")
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text        


def main():
    ratings_dir = r'../ratings/'
    filepaths = []
    for root, dirs, files in os.walk(ratings_dir):
        for filename in files:
            if filename.endswith('txt'):
                filepaths.append(os.path.join(root, filename))

    goal_corpus = create_ratings_corpus(filepaths, 'goal')
    guardian_corpus = create_ratings_corpus(filepaths, 'guardian')

    corpus_file = open('ratings_corpus.json', mode='w')
    json.dump((goal_corpus + guardian_corpus), corpus_file, indent=4)
    corpus_file.close()
    
if __name__ == '__main__':
    main()
