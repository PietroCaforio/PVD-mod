import numpy as np
from collections import Counter
import argparse
import os
import requests
import generate_pointclouds
import objaverse
import pickle


def getElement(e):
        return {
        "name": e["name"],
        "id": e["uid"],
        "image": e["thumbnails"]["images"][0]["url"],
        "description": e["description"],
        "categories": list(map(lambda x: x["name"], e["tags"])),
        "user": e["user"]["username"],
    }

cat_folder = "pointclouds/tags_airplane"
files_list = open(cat_folder + "/files.txt", "r").readlines()
ids = []
files = []
for f in files_list:
    id = os.path.basename(f)
    id = id.split(".")[0]
    ids.append(id)
    files.append(f)
print(ids)

print("Loading meta data of", len(files_list), "objects")

annotations = objaverse.load_annotations(ids)
annotations = list(map(lambda x : getElement(x), annotations.values()))

print("Finished loading metadata")

metadata = open(cat_folder + "/metadata.tsv", "w")
i = 0
added = 0

for i in range(0, len(files)):
    print("Generating poiknt cloud from", files[i], "____", ids[i])
    if generate_pointclouds.save(files[i], ids[i], "tag_airplane", 2048 * 3):
        anno = annotations[i]["categories"]
        added += 1
        def filter(s):
            return s.replace("\t", "").replace("\n", "")
        user = filter(annotations[i]["user"])
        description = filter(annotations[i]["description"])
        model_name = filter(annotations[i]["name"])
        line_anno = filter(" ".join(anno))
        metadata.write(ids[i] + "\t" + line_anno + "\t" + user + "\t" + model_name + "\t" + description + "\n")
    i += 1
    print(i, "/", len(files))
metadata.close()