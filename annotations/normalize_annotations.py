"""
Author: Matthew Garber
Class: Natural Language Annotation for Machine Learning
Term: Spring 2016
Project: SoccEval

This program normalizes certain aspects of the SoccEval annotation so that it
is easier to adjudicate as well as making minor fixes to annotations that were
a result of errors in early version of the DTD. The playerID attribute of
Player and Coref tags are normalized to be as consistent as possible across
the three sets of annotations.

Dependencies:
    bs4
"""

import os
import re
from bs4 import BeautifulSoup as BS
from os.path import join as pjoin

def has_playerid(tag):
    """Returns True if the XML tag has the attribute playerid.

    Args:
        tag: a BeautifulSoup XML tag
    Returns:
        True if tag has playerid, False if otherwise.
    """
    return tag.has_attr('playerid')

def is_unnormalized(tag):
    """Detects whether a given Coref or Player tag has been normalized. All
    normalized tags should have a playerid of 100 or greater.
    
    Args:
        tag: a Player or Coref tag (as a BeautifulSoup object).
    Returns:
        True if tag has a playerid is digits and is less than 100, False if
        otherwise.
    """
    if tag.has_attr('playerid'):
        playerid = tag['playerid']
        return playerid.isdigit() and int(playerid) < 100
    else:
        return False

def change_playerid(annotation, old_id, new_id):
    """Changes the playerid attribute value from old_id to new_id for all tags
    in the given annotation with a playerid equal to old_id.

    Args:
        annotation: a BeautifulSoup parsed XML object.
        old_id: the playerid to change.
        new_id: the new value of the playerid.
    """
    tag_list = annotation.find_all(playerid=old_id)
    for tag in tag_list:
        tag['playerid'] = new_id

def create_base_playerids(annotation):
    """Changes to annotations playerids to a new normalized form that will be
    used as the basis for the normalization of other annotators' annotations.

    Args:
        annotation: a BeautifulSoup parsed XML object.
    """
    playerid_tags = annotation.find_all(has_playerid)
    playerids = set([int(tag['playerid']) for tag in playerid_tags
                     if tag['playerid'].isdigit()]
                    )
    for playerid in playerids:
        tags_with_id = annotation.find_all(playerid=str(playerid))
        for tag in tags_with_id:
            tag['playerid'] = str(playerid + 100)

def normalize_playerids(base_annotation, annotation, offset):
    """Normalizes the playerids of a given annotation based on the playerids of
    tags with matching spans in the base_annotation.

    If two tags in the different sets of annotations both have identical spans
    and have playerids (with the unnormalized tag having a playerid of XX and
    the base tag having a playerid of YY), then all occurrences of playerid
    with value XX are changed to YY.
        If after this there are still playerids
    that have not been changed, then they are increased by the given offset
    (which should be 200 or more so as not to conflict with the base
    annotation.)

    Args:
        base_annotation: a BeautifulSoup parsed XML object used as the basis
            for normalization.
        annotation: a BeautifulSoup parsed XML object storing the unnormalized
            annotation.
        offset: the amount to increase remaining unnormalized tags by.
    """
    anno_tags = annotation.find_all(has_playerid)
    for tag in anno_tags:
        if is_unnormalized(tag):
            match_tag = base_annotation.find(spans=tag['spans'])
            if match_tag:
                old_id = tag['playerid']
                new_id = match_tag['playerid']
                change_playerid(annotation, old_id, new_id)
    remaining_tags = annotation.find_all(is_unnormalized)
    for tag in remaining_tags:
        tag['playerid'] = str(int(tag['playerid']) + offset)

def fix_xml(annotation_text):
    """Fixes syntax errors that resulted from early versions of the DTD.

    Args:
        annotation_text: the XML annotation as a string.
    Returns:
        The annotation_text with the errors corrected.
    """
    text = re.sub('fact-opinion', 'fact_or_opinion', annotation_text)
    text = re.sub('fact or opinion', 'fact_or_opinion', text)
    return text

def apply_changes(parsed_xml, annotation_text):
    """Applies the normalization made in the parsed_xml to the raw
    annotation_text.

    Args:
        parsed_xml: a BeautifulSoup parsed XML object with normalized
            annotations.
        annotation_text: the unnormalized XML annotation as a string.
    Returns:
        The normalized XML annotation as a string.
    """
    text = annotation_text
    for tag, piece in re.findall('(<(Player|Coref).*?>)', annotation_text):
        tag_id = re.search(r'id="(.*?)"', tag).group(1)
        new_player_id = str(parsed_xml.find(id=tag_id)['playerid'])
        pattern = r'(id="'+tag_id+'" .*? playerID=").*?(")'
        text = re.sub(pattern, r'\g<1>'+new_player_id+r'\g<2>', text)
    return text

def main(packages, submissions_dir='Submissions', new_dir='Normalized'):
    annotators = os.listdir(submissions_dir)
    for package in packages:
        for annotator in annotators:
            os.makedirs(pjoin(new_dir, annotator, package))
        files = os.listdir(pjoin(submissions_dir, annotators[2], package))
        for f in files:
            anno1 = open(pjoin(submissions_dir, annotators[2], package, f),
                         encoding='utf8')
            anno2 = open(pjoin(submissions_dir, annotators[1], package, f),
                         encoding='utf8')
            anno3 = open(pjoin(submissions_dir, annotators[0], package, f),
                         encoding='utf8')

            anno1_text = fix_xml(anno1.read())
            anno2_text = fix_xml(anno2.read())
            anno3_text = fix_xml(anno3.read())
            anno1_soup = BS(anno1_text, 'html.parser')
            anno2_soup = BS(anno2_text, 'html.parser')
            anno3_soup = BS(anno3_text, 'html.parser')
                     
            create_base_playerids(anno1_soup)         
            triplets = [(anno1_soup, anno2_soup, 200),
                        (anno1_soup, anno3_soup, 300),
                        (anno2_soup, anno3_soup, 400)]
            for base, other, offset in triplets:
                normalize_playerids(base, other, offset)
                     
            new_anno1 = open(pjoin(new_dir, annotators[0], package, f), 'w',
                             encoding='utf8')
            new_anno2 = open(pjoin(new_dir, annotators[1], package, f), 'w',
                             encoding='utf8')
            new_anno3 = open(pjoin(new_dir, annotators[2], package, f), 'w',
                             encoding='utf8')
            new_anno1.write(apply_changes(anno1_soup, anno1_text))
            new_anno2.write(apply_changes(anno2_soup, anno2_text))
            new_anno3.write(apply_changes(anno3_soup, anno3_text))

if __name__ == '__main__':
    packages = ['Package 1', 'Package 2', 'Package 3', 'Package 4']
    submissions_dir = 'Submissions'
    normalized_dir = 'Normalized'
    main(packages, submissions_dir, normalized_dir)
