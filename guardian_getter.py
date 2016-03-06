import requests, os, re

def tagChecker(dics):
    count = 0
    for d in dics:
        try:
            if d['id']=='football/chelsea': #or d['id']=='football/liverpool' or d['id']=='football/tottenham-hotspur'
                count += 1
            if d['id']=='tone/matchreports':
                count += 1
            if count == 2:
                return True
        except:
            continue
    return False

def getArticles(info, url, directory):
    r = requests.get(url, info)
    while 1:
        r = requests.get(url, info)
        if r.json()["response"]["status"] != "ok":
            break
        else:
            for i in range(r.json()["response"]["pageSize"]):
                print(i)
                #tag = r.json()["response"]["results"][i]["tags"]
                #if not tagChecker(tag):
                    #continue
                
                pubdate = r.json()["response"]["results"][i]["webPublicationDate"]
                pubdate = pubdate.replace(":", "")
                pubdate = pubdate[:-1]
                title = r.json()["response"]["results"][i]["webTitle"]

                if ("Chelsea" not in title and "Liverpool" not in title and "spur" not in title and "Tottenham" not in title):
                    continue
                
                title = title.replace(":", "")
                title = title.replace("*", "")
                title = title.replace("?", "")
                title = title.replace("\"", "")
                title = title.replace("<", "")
                title = title.replace(">", "")
                title = title.replace("|", "")
                title = title.replace("\\", "")
                title = title.replace("\n", "")
                title = title.replace("\r", "")
                title = title.replace("/", "")
                
                if r.json()["response"]["results"][i]["type"] == "article":
                    with open(os.path.join(directory,pubdate+title.strip())+".txt", "w", encoding="UTF-8") as f:
                        f.write(r.json()["response"]["results"][i]["webUrl"])
                        f.write("\n\n")
                        f.write(r.json()["response"]["results"][i]["webTitle"])
                        f.write("\n\n")
                        if "blocks" in r.json()["response"]["results"][i].keys():
                            content = r.json()["response"]["results"][i]["blocks"]["body"][0]["bodyHtml"]
                            content = re.sub(r'</?[ahsbfieutv].*?>','',content)  #<p>, <li> are kept
                            content = re.sub(r'<.*?>','\n',content)
                            f.write(content)
                            f.write("\n\n")
            info["page"] += 1



if __name__ == "__main__":
    info = {"api-key": "test",
            "q": "Chelsea OR Liverpool OR Tottenham OR Spur",#query
            "from-date": "2014-08-10",
            "to-date": "2015-05-31",
            "page-size": 200, #default to show 200 articles on each page
            "page": 1, #default to show first page
            "section": "football",
            "show-blocks": "all",
            "show-tags": "all"}
    url = "http://content.guardianapis.com/search"
    directory = "C://Users//Xinhao//Desktop//titlewithCLT" #dir
    getArticles(info, url, directory)
