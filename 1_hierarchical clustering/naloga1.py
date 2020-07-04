import numpy as np
import pandas as pd
import csv
import math
from statistics import mean
from operator import itemgetter

''' FUNCTIONS 
'''
# row distance or distance between two countries: 
# calculate normalized Euclidean distance between all pairs that are legit (not containing NaN or empty value)
def row_distance(r1, r2):
    diffs = []
    count = 0
    for a,b in zip(r1,r2):
        if (a != None and b != None and not math.isnan(a) and not math.isnan(b)):
            diffs.append((a-b)**2)
            count = count + 1

    if (len(diffs) != 0):
        return (sum(diffs)/len(diffs))**0.5
    else:
        return 0

# flatten a multi-leveled list into single one, for example:
# [1 ,[2, [3, 4]]] => [1, 2, 3, 4]
def flatten(group, flat=[]):
    for e in group:
        if (isinstance(e, list)):
           flatten(e, flat)
        else:
            flat.append(e)

# find preferred and not preferred countries of a group of countries
def find_pref_not_pref(group, countries, data):
    flattened = []
    flatten([group], flattened)

    data = data.fillna(0)

    group_vectors = []
    print("Countries in a group: ", end="")
    for e in flattened:
        print(str(countries[e] + ", "), end="")
        group_vectors.append(data.iloc[e])

    print()
    avg = list(map(mean, zip(*group_vectors)))
    
    sorted_inds, sorted_items = zip(*sorted([(i,e) for i,e in enumerate(avg)], key=itemgetter(1)))

    for i in sorted_inds:
        print(str(data.columns.values[i] + " "), avg[i], ",", end="")
    
    print()
    print()

# calculate the distance between any possible pair of countries from group1 and group2
def cluster_distance(group1, group2, data):
    flattened_group1 = [] # cifra = [1]; array = [[1, [2, 3]]]
    flatten([group1], flattened_group1)

    flattened_group2 = []
    flatten([group2], flattened_group2)

    dst_sum = 0
    for i in flattened_group1:
        for j in flattened_group2:
            dst_sum += row_distance(data.iloc[i], data.iloc[j])

    return dst_sum/(len(flattened_group1)*len(flattened_group2))

# a helper function to print depth of the current level in recursion
def print_depth(count):
    for i in range(count):
        print("    ", end='')

# a recursive function to print a dendrogram
def print_tree(tree, countries, depth = 0):
    if (not isinstance(tree, list)):
        print_depth(depth)
        print("---- " + str(countries[tree]))
    else:
        print_tree(tree[0], countries, depth + 1)
        print_depth(depth)
        print('----|')
        print_tree(tree[1], countries, depth + 1)

# the main algorithm
def hierarchical_clustering(data):
    # get list of countries' names
    countries = list(data.index.values) + list(data.index.values)
    
    #duplicate = list(countries)
    #duplicate = duplicate + duplicate

    # every country is a group
    groups = list(range(0, len(data.index)))

    # while there is at least one group left, do the following:
    while (len(groups) > 1):
        min = np.Inf
        min_i = -1
        min_j = -1

        # find the closest two groups
        for i in range(0, len(groups)):
            for j in range (i+1, len(groups)):
                dist = cluster_distance(groups[i], groups[j], data)
                if (dist < min):
                    min = dist
                    min_i = i
                    min_j = j

        # merge the closest two groups into new_group
        new_group = [groups[min_i], groups[min_j]]

        # find the preferred and not preferred countries of new_group
        #find_pref_not_pref(new_group, countries, data)

        # delete the closest two groups from list of groups
        del groups[min_j]
        del groups[min_i]

        # add new_group to the list of groups
        groups.append(new_group)

    # print dendrogram
    print_tree(groups[0], countries)


''' MAIN
'''
if __name__ == "__main__":
    # read data from each file and save it to Panda dataframe
    data_final = pd.read_csv("data/eurovision-final.csv", encoding = "latin-1", sep = ",", usecols = np.array(list(range(1, 49))))
    data_semifinal = pd.read_csv("data/eurovision-semifinal.csv", encoding = "latin-1", sep = ",", usecols = np.array(list(range(1, 49))))

    # group rows by country and calculate mean
    data_final = data_final.groupby("Country").mean()
    data_semifinal = data_semifinal.groupby("Country").mean()
 
    # transpose to get voting vector
    data_final = data_final.transpose()
    data_semifinal = data_semifinal.transpose()

    # merge both dataframes into one
    data = pd.concat([data_semifinal, data_final], axis=1)
    
    # run algorithm
    hierarchical_clustering(data)