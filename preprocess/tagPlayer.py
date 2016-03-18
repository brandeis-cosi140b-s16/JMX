from operator import itemgetter
import re, os

def inside(num, spans):
    for span in spans:
        if num>=span[0] and num<span[1]:
            return True
    return False

def tagPlayer(doc, players):
    
    with open(doc,"r",-1,"UTF-8") as f:
        content=f.read()

    matchlist = []
    for fn in players:
        m1=re.finditer("\\b"+fn+"\\b", content)
        for match1 in m1:
            matchlist.append([match1.span(),match1.group()])
        m2=re.finditer("\\b"+fn.split()[-1]+"\\b", content)
        for match2 in m2:
            spans = [itemgetter(0)(item) for item in matchlist]
            if not inside(match2.span()[0],spans):
                matchlist.append([match2.span(),match2.group()])

    matchlist=sorted(matchlist, key=itemgetter(0))

    coreflist = []
    m=re.finditer(r"\b[hH]e\b|\b[hH]is\b", content)
    for match in m:
        coreflist.append([match.span(),match.group()])

    coreflist=sorted(coreflist, key=itemgetter(0))

    with open("./"+doc.split("\\")[-1]+".xml","w",-1,"UTF-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8" ?>\n\n<SoccEval1.2>\n<TEXT><![CDATA[""") #v1.2
        f.write(content+"]]></TEXT>\n")
        f.write("<TAGS>\n")
        for i in range(len(matchlist)):
            f.write("""<Player id="%s" spans="%d~%d" text="%s" playerID="null" />\n"""%("P"+str(i),matchlist[i][0][0],matchlist[i][0][1],matchlist[i][1]))
        for i in range(len(coreflist)):
            f.write("""<Coref id="%s" spans="%d~%d" text="%s" playerID="null" />\n"""%("C"+str(i),coreflist[i][0][0],coreflist[i][0][1],coreflist[i][1]))
        f.write("</TAGS>\n</SoccEval1.2>") #v1.2

if __name__ == "__main__":
    with open("./playerlist.txt","r",-1,"UTF-8") as f: #playerlist_noaccent for goal.com, playerlist for guardian
        players = f.read().split("\n")
        players.remove("")
    files=[]
    direc=os.path.dirname(os.path.abspath("./"))+"\\guardian_articles\\seasonreviews"
    for (dirpath, dirnames, filenames) in os.walk(direc):
        files.append([os.path.join(dirpath,name) for name in filenames])
    files=files[0]
    for file in files:
        tagPlayer(file, players)
