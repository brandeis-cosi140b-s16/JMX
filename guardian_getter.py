import requests, os, re

info = {"api-key": "test",
        "q": "Chelsea",
        "from-date": "2014-08-10",
        "to-date": "2015-05-31",
        "page-size": 200, #default to show 200 articles on each page
        "page": 1, #default to show first page
        "section": "football",
        "show-blocks": "all"}

url = "http://content.guardianapis.com/search"

r = requests.get(url, info)
directory = "/directory"  #hard-coded

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
            if r.json()["response"]["results"][i]["type"] == "article":
                with open(os.path.join(directory,pubdate+title[0])+".txt", "w") as f:
                    
                    """1)have to fix encoding problem, 2)have to preprocess html"""
                    
                    print(r.json()["response"]["results"][i]["webUrl"].encode('utf-8','ignore'), file=f)
                    f.write("\n\n")
                    print(r.json()["response"]["results"][i]["webTitle"].encode('utf-8','ignore'), file=f)
                    f.write("\n\n")

                    if "blocks" in r.json()["response"]["results"][i].keys():
                        content = r.json()["response"]["results"][i]["blocks"]["body"][0]["bodyHtml"]
                        content = re.sub(r'</?[ahsbfieutv].*?>','',content)  #<p>, <li> are kept
                        print(content.encode('utf-8','ignore'), file=f)
                
        info["page"] += 1

