import numpy as np
from collections import Counter
import argparse
import os
import requests
import generate_pointclouds
import objaverse
import pickle

# Script to load models from objaverse and convert them to a pointcloud

def getObja():
    if os.path.exists("obja.pkl"):
        print("Downloading annotations")
        obja = objaverse.load_annotations()
        with open("obja.pkl", 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obja, outp, pickle.HIGHEST_PROTOCOL)
        print("Finished downloading annotations")
        return obja
    else:
        print("Reading existing annotations")
        with open("obja.pkl", 'r') as f:
            return pickle.load(f)

#annotations = objaverse.load_annotations(uids[:10])
#annotations = objaverse.load_annotations()
#cc_by_uids = [uid for uid, annotation in annotations.items() if annotation["license"] == "by"]

parser = argparse.ArgumentParser(description="A script to handle removal of items from a list.")
# Adding the 'remove' argument which expects a list of strings
parser.add_argument('--remove', nargs='+',type=str,help='A list of strings to be removed.')
parser.add_argument('--cat',nargs='+',type=str,default=[],help='A list with at least one string for categorization. Default is [].')
parser.add_argument('--tags',nargs='+',type=str,default=None,help='A list with at least one string for tags. Default is None.')
parser.add_argument('--points', type=int, default=(2048 * 3), help='Number of points (default: 2048*3)')
parser.add_argument('--max', type=int, default=10000000000, help='Number of models (default: 150)')
args = parser.parse_args()
points = args.points
maximum = args.max
banned_authors = []
if args.remove:
    banned_authors += list(args.remove)
    print("Ignore authors: ", banned_authors)

def getCategoryList(cat):
    return list(map(lambda x: x["name"], cat))

def getElement(e):
        return {
        "name": e["name"],
        "id": e["uid"],
        "image": e["thumbnails"]["images"][0]["url"],
        "description": e["description"],
        "categories": list(map(lambda x: x["name"], e["tags"])),
        "user": e["user"]["username"],
    }
    
f = open("stats.txt", "w")
print("len points: ", points)

def out(*args ):
    result = ' '.join(str(arg) for arg in args) + "\n"
    f.write(result)
def analyze_category(c):
    n_cats = []
    authors = []
    all_cats = []
    for object in c:
        n_cats.append(len(object["categories"]))
        authors.append(object["user"])
        all_cats = all_cats + object["categories"]
    #print(n_cats)
    all_cats = dict(Counter(all_cats))
    unique_cat_count = []
    for object in c:
        unique = 0
        for cat in object["categories"]:
            if all_cats[cat] == 1:
                unique += 1
        unique_cat_count.append(unique)
    
    is_unique = len([num for num in unique_cat_count if num > 0])

    out("Total no. of tags  : ", len(all_cats))
    out("Objects with unique tags:", is_unique, "/", len(unique_cat_count))
    out("Mean no. of tags:", np.mean(n_cats))
    out("Median no. of tags:", np.median(n_cats))
    authors = dict(Counter(authors))
    out("No. of authors: ", len(authors))
    out("Mean no. of authors models:", np.mean(list(authors.values())))
    out("Median no. of authors models:", np.median(list(authors.values())))
    
emergency = False
if emergency:    
    lines = open("pointclouds/" + "chair" + "/metadata.tsv", encoding="utf-8").readlines()
    filtered = list(map(lambda x : x.split("\t")[0], lines))
elif args.tags is not None:
    argsset = list(args.tags)
    name = str("tags_" + "_".join(args.tags))
    all_annotations = getObja()
    print(set(map(lambda x : x["name"],list(all_annotations.items())[0][1]["tags"])))
    filtered = [uid for uid, annotation in all_annotations.items() if not(argsset[0] in set(map(lambda x : x["name"],annotation["tags"])))]
elif args.cat is not None:
    name = "cat_" + args.cat[0]
    lvis_annotations = objaverse.load_lvis_annotations()
    filtered = lvis_annotations[args.cat[0]]
else:
    print("Error, need either --cat or --tags")
    exit()


annotations = objaverse.load_annotations(filtered)
annotations = list(map(lambda x : getElement(x), annotations.values()))
annotations = list(filter(lambda e : len(e["categories"])>=3, annotations))
print("Before removing blacklisted authors: ", len(annotations))
annotations = list(filter(lambda x : x["user"] not in banned_authors, annotations))
print("After removing blacklisted authors: ", len(annotations))
#annotations = annotations[:maximum + 1000]
#analyze_category(annotations)
out("")
if not emergency:
    print("Start objects loading")
    import multiprocessing
    cpus = multiprocessing.cpu_count()
    res = objaverse.load_objects(list(map(lambda x : x["id"], annotations))[:maximum], min(cpus, 8))
    print("Finish objects loading")
else:
    res = dict(map(lambda x : (x, "glbs/" + x + ".glb"), filtered))
    name = "chair"
print(len(res), "objets")

# for anno in annotations:
#     url = anno["image"]
#     response = requests.get(url)
#     with open("images/" + anno["id"] + ".jpeg", 'wb') as file:
#         file.write(response.content)
#         print("Wrote to file")
cat_folder = "pointclouds/" + name
if not os.path.exists(cat_folder):
    os.makedirs(cat_folder)
files_list = open(cat_folder + "/files.txt", "w")
    
metadata = open(cat_folder + "/metadata.tsv", "w")
files = list(res.items())

for file in files:
    files_list.write(file[1] + "\n")
files_list.close()

i = 0
added = 0

for file in files:
    print("Generating poiknt cloud from", file[1], "____", file[0])
    if generate_pointclouds.save(file[1], file[0], name, points):
        anno = annotations[i]["categories"]
        added += 1
        def filter(s):
            return s.replace("\t", "").replace("\n", "")
        user = filter(annotations[i]["user"])
        description = filter(annotations[i]["description"])
        model_name = filter(annotations[i]["name"])
        line_anno = filter(" ".join(anno))
        metadata.write(file[0] + "\t" + line_anno + "\t" + user + "\t" + model_name + "\t" + description + "\n")
    i += 1
    print(i, "/", len(files))
    if added > maximum:
        break
metadata.close()
        
