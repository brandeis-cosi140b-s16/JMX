import requests, os

#BUGS to fix:
#1 not sure why there's list out of range error
#2 encoding problem to fix
#3 preprocess html

info = {"api-key": "__",
        "q": "Chelsea",
        "from-date": "2014-08-16",
        "to-date": "2014-09-15",
        "page-size": 200, #default to show 200 articles on each page
        "page": 1, #default to show first page
        "section": "football",
        "show-blocks": "all"}

url = "http://content.guardianapis.com/search"

r = requests.get(url, info)
directory = "C:/testgitrepo"  #hard-coded


while 1:
    r = requests.get(url, info)
    if r.json()["response"]["status"] != "ok":
        break
    else:
        for i in range(r.json()["response"]["pageSize"]):
            print(i)
            pubdate = r.json()["response"]["results"][i]["webPublicationDate"]
            pubdate = pubdate.replace(":", "")
            title = r.json()["response"]["results"][i]["webTitle"]  #have to fix encoding problem
            title = title.replace(":", "")
            with open(os.path.join(directory,pubdate+title[0])+".txt", "w") as f:
                
                """1)have to fix encoding problem, 2)have to preprocess html"""
                
                print(r.json()["response"]["results"][i]["webUrl"].encode('utf-8','ignore'), file=f)
                f.write("\n\n")
                print(r.json()["response"]["results"][i]["webTitle"].encode('utf-8','ignore'), file=f)
                f.write("\n\n")
                try:
                    print(r.json()["response"]["results"][i]["blocks"]["body"][0]["bodyHtml"].encode('utf-8','ignore'), file=f)
                except KeyError:
                    pass
                
        info["page"] += 1

