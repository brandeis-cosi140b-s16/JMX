from bs4 import BeautifulSoup
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys, os, itertools
import numpy as np
from sklearn import linear_model,ensemble,svm
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import pearsonr

STOPWORDS = stopwords.words('english')
STOPWORDS.extend([',', '.'])
ALL_RATINGS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

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
            elif tag['polarity']=='-':
                total -= 1
    return total

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


def get_playerid_rating_dic(filename, roots="../gold_standard"):
    """roots should be gold_standard root, filename is the name of the xml file"""
    xml=open(os.path.join(roots, filename), encoding="UTF-8").read()
    bs = BeautifulSoup(xml, 'html.parser')

    date=filename.replace("-","")[:8]
    rating_folder="../ratings/"

    #find corresponding rating file
    rating_file=""
    for root, folder, file in os.walk(rating_folder):
        for f in file:
            if f.startswith("r"+date) and "chelsea" in f:
                rating_file=os.path.join(root,f)
                break
    
    #parse rating into a dictionary
    ratings=parse_ratings(open(rating_file))

    #link playerid to rating, "-1" means player rating not found in file
    playerid_rating_dic={}
    for player_tag in bs.find_all("player"):
        name=player_tag["text"].lower()
        name=strip_accents(name)
        has_rating=False
        for d in ratings.values():
            if len(name.split())>1:
                if d["names"][0].split("\\b")[1]==name or d["names"][2].split("\\b")[1]==name.split()[1:] or d["names"][1].split("\\b")[1]==name.split()[0]:
                    playerid_rating_dic[player_tag["playerid"]]=[d["rating"],name]
                    has_rating=True
            elif len(name.split())==1:
                if d["names"][0].split("\\b")[1]==name or d["names"][2].split("\\b")[1]==name or d["names"][1].split("\\b")[1]==name:
                    playerid_rating_dic[player_tag["playerid"]]=[d["rating"],name]
                    has_rating=True
        if has_rating==False:
            playerid_rating_dic[player_tag["playerid"]]=[-1,name]
    return playerid_rating_dic

def for_ref(data):
    with open("rawdata_for_ref.csv","w", encoding="UTF-8", newline='') as f:
        wr=csv.writer(f,delimiter=",",quoting=csv.QUOTE_ALL)
        for l in data:
            wr.writerow([str(i) for i in l])

def create_rawset(roots, filename):
    if "review" not in filename:
        xml=open(os.path.join(roots, filename), "r", -1, "UTF-8").read()
        bs=BeautifulSoup(xml, 'html.parser')
        player_links = map_playerids_to_targets(bs)
        player_ratings=get_playerid_rating_dic(filename,roots)
        l=[[filename.replace("-","")[:8]+"-"+pid, player_links[pid],
            player_ratings[pid][0], player_ratings[pid][1]] for pid in player_links.keys()]
        return l

def doc_tag_counts(train_raw):
    """return dictionary mapping docid to tag total counts"""
    d={}
    for l in train_raw:
        if l[2]!=-1:
            d[l[0][:8]]=d.get(l[0][:8],0)+len(l[1])
    return d

##features
def create_X(train_raw):
    doc_tags=doc_tag_counts(train_raw)
    train_X=[]
    for l in train_raw:
        if l[2]!=-1:
            feature_list=[]
            #fact
            feature_list.append(sum(1 for t in l[1] if t.name=='fact' and t['type']=="goal")/doc_tags[l[0][:8]])#goal
            feature_list.append(sum(1 for t in l[1] if t.name=='fact' and (t['type']=="assist" or t['type']=="pass"))/doc_tags[l[0][:8]])#assist/pass
            feature_list.append(sum(1 for t in l[1] if t.name=='fact' and t['type']=="shot")/doc_tags[l[0][:8]])#shot
            feature_list.append(sum(1 for t in l[1] if t.name=='fact' and (t['type']=="movement" or t['type']=="positioning"))/doc_tags[l[0][:8]])#movement/positioning
            feature_list.append(sum(1 for t in l[1] if t.name=='fact' and t['type']=="substitute out")/doc_tags[l[0][:8]])#sub out
            feature_list.append(sum(1 for t in l[1] if t.name=='fact' and t['type']=="substitute in")/doc_tags[l[0][:8]])#sub in
            feature_list.append(sum(1 for t in l[1] if t.name=='fact' and t['type']=="injury")/doc_tags[l[0][:8]])#injury
            feature_list.append(sum(1 for t in l[1] if t.name=='fact' and t['type']=="tackle")/doc_tags[l[0][:8]])#tackle
            feature_list.append(sum(1 for t in l[1] if t.name=='fact' and t['type']=="save")/doc_tags[l[0][:8]])#save
            feature_list.append(sum(1 for t in l[1] if t.name=='fact' and t['type']=="foul")/doc_tags[l[0][:8]])#foul

            #opinion
            feature_list.append(total_polarity(l[1]))
            feature_list.append(sum(1 for t in l[1] if t.name=='opinion' and t['type']=="soccer skill")/doc_tags[l[0][:8]])#soccer skill
            feature_list.append(sum(1 for t in l[1] if t.name=='opinion' and (t['type']=="accomplishment" or t['type']=="growth-decline"))/doc_tags[l[0][:8]])#accomplishment/growth-decline
            feature_list.append(sum(1 for t in l[1] if t.name=='opinion' and t['type']=="general attribute")/doc_tags[l[0][:8]])#general attribute
            feature_list.append(sum(1 for t in l[1] if t.name=='opinion' and t['type']=="impact on team")/doc_tags[l[0][:8]])#impact on team
            feature_list.append(sum(1 for t in l[1] if t.name=='opinion' and t['type']=="other opinion")/doc_tags[l[0][:8]])#other opinion
            
            #fact or opinion total count
            feature_list.append(len(l[1]))
            
            train_X.append(feature_list)
    return train_X

def rating_convert(num_list):
    return np.asarray([ALL_RATINGS[list(np.absolute([i-num for i in ALL_RATINGS])).index(
        min(list(np.absolute([i-num for i in ALL_RATINGS]))))] for num in num_list])

def getallfiles(folder="../gold_standard/"):
    allfiles=[]
    for roots, folders, files in os.walk("../gold_standard/"):
        for filename in files:
            allfiles.append(filename)
    return allfiles

def main(allfiles,leaveout,show_result=True,show_correlate=True):
#    allfiles=[]
    train_raw=[] #used for feature extraction
    test_raw=[] #used for feature extraction
#    for roots, folders, files in os.walk("../gold_standard/"):
#        for filename in files:
#            allfiles.append(filename)
            
    train=allfiles[:leaveout]+allfiles[leaveout+2:]
    test=allfiles[leaveout:leaveout+2]

    for f in train:
        if "review" not in f:
            train_raw.extend(create_rawset("../gold_standard/",f))
    for f in test:
        if "review" not in f:
            test_raw.extend(create_rawset("../gold_standard/",f))

    train_X=create_X(train_raw)
    train_X=np.asarray(train_X)
    train_y=[l[2] for l in train_raw if l[2]!=-1]
    train_y=np.asarray(train_y)

    test_X=create_X(test_raw)
    test_X=np.asarray(test_X)
    test_y=[l[2] for l in test_raw if l[2]!=-1]
    test_y=np.asarray(test_y)

    #linear regression
    regr=linear_model.LinearRegression()
    regr.fit(train_X, train_y)

    pred=rating_convert(regr.predict(test_X))
    result_lr=precision_recall_fscore_support(test_y.astype(str), pred.astype(str), average='micro')[:-1]
    if show_result:
        print("LR: %5s %5s %5s" % ("P","R","F"))
        print("micro %4.2f, %4.2f, %4.2f" % result_lr)

    #svr
    svr=svm.SVR()
    svr.fit(train_X,train_y)
    pred=rating_convert(regr.predict(test_X))
    result_svr=precision_recall_fscore_support(test_y.astype(str), pred.astype(str), average='micro')[:-1]
    if show_result:
        print("SVR:%5s %5s %5s" % ("P","R","F"))
        print("micro %4.2f, %4.2f, %4.2f" % result_svr)

    #maxent
    maxent=linear_model.LogisticRegression()
    maxent.fit(train_X,train_y.astype(str))
    pred=maxent.predict(test_X)
    result_maxent=precision_recall_fscore_support(test_y.astype(str), pred.astype(str), average='micro')[:-1]
    if show_result:
        print("ME: %5s %5s %5s" % ("P","R","F"))
        print("micro %4.2f, %4.2f, %4.2f" % result_maxent)

    #random forest
    rf=ensemble.RandomForestClassifier()
    rf.fit(train_X,train_y.astype(str))
    pred=rf.predict(test_X)
    result_rf=precision_recall_fscore_support(test_y.astype(str), pred.astype(str), average='micro')[:-1]
    if show_result:
        print("RF: %5s %5s %5s" % ("P","R","F"))
        print("micro %4.2f, %4.2f, %4.2f" % result_rf)

    if show_correlate:
        print("\ncorrelated features with sig:")
        for i,j in itertools.combinations(range(train_X.shape[1]),2):
            if pearsonr(train_X[:,i],train_X[:,j])[1]<0.001:
                print(i,j,pearsonr(train_X[:,i],train_X[:,j]))
        print()
    return result_lr, result_svr, result_maxent, result_rf

if __name__=="__main__":
    sys.path.append("../baseline")
    from create_baseline_corpus import parse_ratings, strip_accents
    results_lr=[]
    results_svr=[]
    results_maxent=[]
    results_rf=[]
    itr=33
    for i in range(itr):
        print("iteration "+str(i+1)+"/"+str(itr)+"\n")
        l=main(getallfiles(),i,show_result=0,show_correlate=0)
        results_lr.append(l[0])
        results_svr.append(l[1])
        results_maxent.append(l[2])
        results_rf.append(l[3])
    print("LR average:")
    print(np.mean(np.asarray(results_lr),0))
    print("SVR average:")
    print(np.mean(np.asarray(results_svr),0))
    print("MaxEnt average:")
    print(np.mean(np.asarray(results_maxent),0))
    print("RF average:")
    print(np.mean(np.asarray(results_rf),0))

#Cross-Validation (leave out 2 test files each time)
#LR average:
#[ 0.36187642  0.36187642  0.36187642]
#SVR average:
#[ 0.36187642  0.36187642  0.36187642]
#MaxEnt average:
#[ 0.36371296  0.36371296  0.36371296]
#RF average:
#[ 0.27971995  0.27971995  0.27971995]
