"""source: http://www.transfermarkt.co.uk/chelsea-fc/startseite/verein/631?saison_id=2014"""

import os

def preNormalize(filename):
    f1=open("./"+filename+".txt", "r", -1, "UTF-8")
    f2=open("./"+filename+"players"+".txt", "w", -1, "UTF-8")
    while 1:
        l1=f1.readline()
        l2=f1.readline()
        l3=f1.readline()
        l4=f1.readline()
        if len(l4.split("\t"))==2:
            l4=l4.strip()+f1.readline().strip()
        if not l1:
            break
        f2.write(l1.strip()+"\t"+l2.split("\t")[0]+"\t"+l3.strip()+l4.strip()+"\n")
    f1.close()
    f2.close()

def getPlayerList(direc):
    playerlist=[]
    files=[]
    for (dirpath, dirnames, filenames) in os.walk(direc):
        files.extend(os.path.join(dirpath,name) for name in filenames)
    for file in files:
        with open(file, 'r', -1, 'UTF-8') as f:
            for line in f:
                playerlist.append(line.split("\t")[1])
    return playerlist

if __name__ == "__main__":
    direc="./playersinteams"
    l=getPlayerList(direc)
    l=list(set(l))
    #print(l)
    
    with open("./playerlist.txt", "w", -1, "UTF-8") as f:
        for player in l:
            f.write(player+"\n")
    
