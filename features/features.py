from bs4 import BeautifulSoup
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


STOPWORDS = stopwords.words('english')
STOPWORDS.extend([',', '.'])

# merged_ types = dict of merged opinion types

FACT_TYPES = ['goal', 'assist', 'pass', 'shot', 'movement', 'positioning',
              'substitute out', 'substitute in', 'injury', 'tackle', 'save',
              'foul']

def map_playerids_to_targets(xml):
    player_links = {}
    for target_link in xml.find_all(name='targetlink'):
        player_tag = xml.find(attrs={'id': target_link['targetid']})
        fact_opin_tag = xml.find(attrs={'id': target_link['fact_or_opinionid']})

        player_id = player_tag['playerid']
        player_links[player_id] = player_links.get(player_id, [])
        player_links[player_id].append(fact_opin_tag)
    return player_links

#### Feature Functions ####

def fact_mentions(player_tag_list):
    """Counts the number of times a Fact is mentioned, regardless of whether
    or not it is repeated.
    """
    features = {}
    for tag in player_tag_list:
        if tag.name == 'fact':
            feature_name = tag['type'] + '_mention'
            features[feature_name] = features.get(feature_name, 0) + 1
    return features

def fact_unique(player_tag_list):
    """Counts the number of unique Facts (i.e. only counts Facts once per
    factID.
    """
    features = {}
    seen_ids = []
    for tag in player_tag_list:
        if tag.name == 'fact':
            if tag['factid'] not in seen_ids:
                feature_name = tag['type'] + '_unique'
                features[feature_name] = features.get(feature_name, 0) + 1
                seen_ids.append(tag['factid'])
    return features

def opin_type_and_polarity(player_tag_list):
    """Creates dict of features mapping opinion type, polarity pairs to a
    Boolean. For example 'soccer skill-' means there was a negative soccer skill
    Opinion.
    """
    features = {}
    for tag in player_tag_list:
        if tag.name == 'opinion':
            # merged_type = merged_types.get(tag['type'], tag['type'])
            # feature_name =merged_type + tag['polarity']
            feature_name = tag['type'] + tag['polarity']
            features[feature_name] = True
    return features

def total_polarity(player_tag_list):
    """Sums polarity over all Opinions (+1 for pos, -1 for neg).
    """
    total = 0
    for tag in player_tag_list:
        if tag.name == 'opinion':
            if tag['polarity']  == '+':
                total += 1
            else:
                total -= 1
    return {'total_polarity': total}

def player_mentions(player_id, parsed_xml):
    """Counts the number of times a player was mentioned with either a Player
    or Coref tag.
    """
    mentions = len(parsed_xml.find_all(attrs={'playerid': player_id}))
    return {'player_mentions': mentions}

#--- Unigram Annotation Features ---#

def tag_unigram(player_tag_list, tag_name):
    """Returns Boolean unigram features for a given class of tag.
    """
    unigrams = FreqDist()
    for tag in player_tag_list:
        if tag.name == tag_name:
            tokens = [token.lower() for token in word_tokenize(tag['text'])
                      if token.lower() not in STOPWORDS]
            for token in tokens:
                unigrams[token] += 1
    return unigrams

###########################

def example():
    xml = open('../gold_standard/2014-08-18T210900Burnley 1-3 Chelsea  Premier League match report_Gold.xml', encoding='utf8').read()
    bs = BeautifulSoup(xml, 'html.parser')
    player_links = map_playerids_to_targets(bs)
    player_id = '103'
    features = fact_mentions(player_links[player_id])
    print(features)
    features = fact_unique(player_links[player_id])
    print(features)
    features = opin_type_and_polarity(player_links[player_id])
    print(features)
    features = total_polarity(player_links[player_id])
    print(features)
    features = player_mentions(player_id, bs)
    print(features)
    features = tag_unigram(player_links[player_id], 'fact')
    print(features.pprint())
    features = tag_unigram(player_links[player_id], 'opinion')
    print(features.pprint())
    features = tag_unigram(player_links[player_id], 'coref')
    print(features.pprint())
