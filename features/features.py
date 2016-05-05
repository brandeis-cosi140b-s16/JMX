from bs4 import BeautifulSoup

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
    features = {}
    for tag in player_tag_list:
        if tag.name == 'fact':
            feat_name = tag['type'] + '_mention'
            features[feat_name] = features.get(feat_name, 0) + 1
    return features

###########################

def example():
    xml = open('../gold_standard/2014-08-18T210900Burnley 1-3 Chelsea  Premier League match report_Gold.xml', encoding='utf8').read()
    bs = BeautifulSoup(xml, 'html.parser')
    player_links = map_playerids_to_targets(bs)
    features = fact_mentions(player_links['103'])
    print(features)
