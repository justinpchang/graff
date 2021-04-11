import numpy as np
import time
import glob
import os.path
import json
from annoy import AnnoyIndex
from scipy import spatial


def cluster():
    print("generating annoy index")

    file_index_to_file_name = {}
    file_index_to_file_vector = {}

    dims = 1280
    n_nearest_neighbors = 2
    trees = 10

    allfiles = glob.glob("./assets/vectors/*.npz")
    t = AnnoyIndex(dims, metric="angular")

    for file_index, file in enumerate(allfiles):
        file_vector = np.loadtxt(file)
        file_name = os.path.basename(file).split('.')[0]
        file_index_to_file_name[file_index] = file_name
        file_index_to_file_vector[file_index] = file_vector

        t.add_item(file_index, file_vector)

        print("\tannoy index %s" %file_name)

    t.build(trees)

    print("calculating similarity score")

    named_nearest_neighbors = []

    for i in file_index_to_file_name.keys():
        master_file_name = file_index_to_file_name[i]
        master_file_vector = file_index_to_file_vector[i]
        
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)

        for j in nearest_neighbors:
            neighbor_file_name = file_index_to_file_name[j]
            neighbor_file_vector = file_index_to_file_vector[j]
            
            similarity = 1 - spatial.distance.cosine(master_file_vector, neighbor_file_vector)
            rounded_similarity = int((similarity * 10000)) / 10000.0

            named_nearest_neighbors.append({
                "similarity": rounded_similarity,
                "master_file_name": master_file_name,
                "similar_file_name": neighbor_file_name,
            })
    
    print("dumping into nearest_neighbors.json")

    with open("nearest_neighbors.json", "w") as out:
        json.dump(named_nearest_neighbors, out)


cluster()