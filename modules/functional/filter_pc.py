import point_cloud_utils as pcu
import numpy as np
import random
import os, shutil

# Python script to filter out the elements with highest average distance from the rest of the dataset

folder = "chair"
count = 50

def normalize_pc(points):
	centroid = np.mean(points, axis=0)
	points -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	points /= furthest_distance
	return points


metadata = open("pointclouds/" + folder + "/metadata.tsv", encoding="utf-8")
lines = metadata.readlines()
res = {}
s = 0
print("Found", len(lines), "objects")
for line in lines:
    if (line == "\n"):
        continue
    cols = line.split("\t")
    name = cols[0]
    res[name] = cols
    s += 1
    if s % 10 == 0:
        print(s, "/", len(lines))

clouds = {}
print("Loading models and normalizing...")
s = 0
for model in res.keys():
    cloud = np.load("pointclouds/" + folder + "/" + model + ".npy")
    clouds[model] = normalize_pc(cloud)
    np.save("pointclouds/" + folder + "/" + model + ".npy", cloud)
    s += 1
    if s % 10 == 0:
        print(s, "/", len(lines))

count = min(count, len(clouds))
cloudsarray = list(clouds.values())
print("Finished loading", len(clouds), "models")
print("Starting average calculations")
progress = 0
averages = {}
for model in res.keys():
    this = clouds[model]
    dists = np.zeros(count)
    for i in range(0, count):
        other =  cloudsarray[random.randint(0, len(res)-1)]
        dist = pcu.hausdorff_distance(this, other)
        #dist = pcu.earth_movers_distance(this, other)
        
        # M = pcu.pairwise_distances(this, other)
        # w_a = np.ones(this.shape[0])
        # w_b = np.ones(other.shape[0])
        # P = pcu.sinkhorn(w_a, w_b, M, eps=1e-7)
        # dist = (M*P).sum()
        
        dists[i] = dist
    average = dists.mean()
    progress += 1
    if progress % 10 == 0:
        print(progress, "/", len(res))
    averages[model] = average
sorted = list(sorted(averages.items(), key=lambda item: item[1]))
print(len(sorted))
file = "pointclouds/" + folder + "/filtered/"
os.makedirs(file, exist_ok=True)
new_metadata = open(file + "metadata.tsv", encoding="utf-8", mode="w")
for i in range(0, int(0.85 * len(sorted))):
    new_metadata.write("\t".join(res[sorted[i][0]]))