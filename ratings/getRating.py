import re, os
from urllib import request

def getRatings(direc):
    files=[]
    url=""

    for (dirpath, dirnames, filenames) in os.walk(direc):
        files.extend(os.path.join(dirpath,name) for name in filenames)

    for file in files:
        with open(file, 'r', -1, 'UTF-8') as f:
            url=f.readline()
        url=url.strip()
        url+="/ratings"
        url="http:"+url.split(":")[1]
        print("fetching data from... "+url)
        content=request.urlopen(url)
        raw=content.read().decode()
        raw=raw[raw.find("Team Ratings"):]
        li=re.findall(r"\<li\>\s*\<a\shref=.+?\<span\sdata-rate=.+?\<\/li\>",raw.replace("\n",""))
        with open(".\\r"+file.split("\\")[-1],"w", -1, "UTF-8") as f:
            for l in li:
                f.write(re.findall(r"\shref=.+?\s" ,l)[0].strip()+" "+re.findall(r"\sdata-rate=.+?\s",l)[0].strip()+"\n")

if __name__ == "__main__":
    direc="..\\goal_articles\\"
    getRatings(direc)
