import sys
import math
import random
import itertools
import operator
from unidecode import unidecode
import seaborn as sns
import matplotlib.pyplot as plt

''' FUNCTIONS 
'''
# return a vector (dictionary of 3-letter substrings) out of a given language
def buildVector(languagePath):
    dict = {}
    file = open(languagePath, "rt", encoding="utf8")
    content = unidecode(file.read())
    for word in content.split():
        word_len = len(word)
        for i in range(0, word_len):
            if (i+3 <= word_len):
                three = word[i:i+3].lower()
                if (three in dict):
                    dict[three] = dict[three] + 1
                else:
                    dict[three] = 1
            else:
                break

    return dict

# return vector length
def vectorLength(v):
    d = 0
    for key in v:
        d += v[key]**2

    return math.sqrt(d)

# return vector product
def vectorProduct(v1, v2):
    p = 0
    for key in v1:
        if (key in v2):
            p += v1[key] * v2[key]
    
    return p

# return cosine similarity between two vectors
def simCosine(v1, v2):
    return (vectorProduct(v1, v2) / (vectorLength(v1) * vectorLength(v2)))

# assign each language to the nearest medoid
def assignToMedoid (medoids_idcs, languages):
    dict_lan_med = {}
    for l in range(0, len(languages)):
            d = float("-inf")
            for m in medoids_idcs:
                sim = simCosine(languages[l], languages[m])
                if (sim > d):
                    d = sim
                    belong_to = m
            dict_lan_med[l] = belong_to

    return dict_lan_med

# find new medoids of groups based on error
def findNewMedoids(medoids_idcs, dict_lan_med, languages):
    new_medoids_idcs = []
    for m in medoids_idcs:
        group = [k for k, v in dict_lan_med.items() if v == m]
        pairs = list(itertools.combinations(group, 2))
        # calculate similarity between all pairs
        dict_pair_value = {}
        for p in pairs:
                dict_pair_value[p] = simCosine(languages[p[0]], languages[p[1]])
        # find new medoid of a group based on error
        min_err = float("inf")
        new_medoid = m
        for i in group:
            err = 0
            for p in pairs:
                if i in p:
                    err += (1 - dict_pair_value[p])
            if (err < min_err):
                min_err = err
                new_medoid = i
        new_medoids_idcs.append(new_medoid)

    return new_medoids_idcs

# return distance to the nearest other group of given language
def findNearestGroup(m, g, medoids_idcs, dict_lan_med, languages):
    min_dist = float("inf")
    nearest_group = m
    for n in medoids_idcs:
        if (n==m):
            continue
        other_group_members = [k for k, v in dict_lan_med.items() if v == n]
        dist = 0.0
        for og in other_group_members:
            dist += (1 - simCosine(languages[g], languages[og]))
        dist = dist / len(other_group_members)
        if (dist < min_dist):
            min_dist = dist
            nearest_group = n
    return min_dist

# return silhouette
def silhouette(medoids_idcs, dict_lan_med, languages):
    sil_all = 0
    for m in medoids_idcs:
        group_members = [k for k, v in dict_lan_med.items() if v == m]
        pairs = list(itertools.combinations(group_members, 2))
        # calculate similarity between all pairs
        dict_pair_value = {}
        for p in pairs:
            dict_pair_value[p] = simCosine(languages[p[0]], languages[p[1]])
        group_sillhouetes = []
        # calculate silhouette of every language in a group and add to sum
        for g in group_members:
            a_g = 0.0
            for p in pairs:
                if g in p:
                    a_g += (1 - dict_pair_value[p])
            a_g = a_g / len(group_members)
            b_g = findNearestGroup(m, g, medoids_idcs, dict_lan_med, languages)
            s = (b_g - a_g) / max(a_g, b_g)
            sil_all += s
        
    return sil_all/len(languages)

# k-medoids clustering algorithm
def medoids_clustering(k, languages, language_names):
    # randomly choose k languages as starting medoids
    medoids_idcs = random.sample(range(len(languages)), 5)
    
    # run algorithm until medoids stay the same
    while True:
        # assign each language to the nearest medoid
        dict_lan_med = assignToMedoid(medoids_idcs, languages)
                
        # calculate new medoids
        new_medoids_idcs = findNewMedoids(medoids_idcs, dict_lan_med, languages)

        if (set(medoids_idcs)==set(new_medoids_idcs)):
            break

        medoids_idcs = new_medoids_idcs

    # calculate silhouette
    s = silhouette(medoids_idcs, dict_lan_med, languages)
    
    return s, medoids_idcs, dict_lan_med

def k_medoids_clustering(languages, language_names_long):
    sil = []
    max_sil = float("-inf")
    learning_times = 100
    num_medoids = 5
    # run algorithm learning_times times and save each silhouette to list
    for i in range(0, learning_times):
        print("{}/{}".format(i+1, learning_times))
        s, medoids_idcs, dict_lan_med = medoids_clustering(num_medoids, languages, language_names_long)
        sil.append(s)

        if (s > max_sil):
            max_sil = s
            best_medoids_idcs = medoids_idcs
            best_dict_lan_med = dict_lan_med

    sys.stdout.flush()
    # print best group arrangement
    for m in best_medoids_idcs:
        group_members = [k for k, v in best_dict_lan_med.items() if v == m]
        print("{", end="")
        for g in group_members:
            print(language_names_long[g], " ", end="")
        print("}")

    # sort silhouettes in descending order
    sil = sorted(sil, reverse=True)
    
    y_labels = []
    for i in range(0, learning_times):
        y_labels.append(i)
    
    # draw barplot
    sns.set(style="whitegrid")
    sns.set_color_codes("pastel")
    sns.barplot(x=sil, y=y_labels, color="b", orient="h")

    plt.show()

# find out in what language the given text is written in
# return a list of three most probable languages
def find_out_language(text_path, languages, language_names_long):
    # build vector out of given text
    v_text = buildVector(text_path)
    
    # list of similarity values between given text and each language
    dict_lan_sim = {}
    for l in range(0, len(languages)):
        dict_lan_sim[language_names_long[l]] = simCosine(v_text, languages[l])

    sorted_dict = sorted(dict_lan_sim.items(), key=operator.itemgetter(1))
    sorted_dict.reverse()

    print(text_path)
    print("My algo thinks it is: ", end="")

    for i in range (0,3):
        print(sorted_dict[i], end="")
    print()
    print()
    
    return sorted_dict[:3]

def finding_out_language(languages):
    texts_names = ["cro", "dns", "eng", "frn", "ger", "itn", "rus", "slv", "spn", "swd"]
    text_names_long = ["croatian", "danish", "english", "french", "german", "italian", "russian", "slovenian", "spanish", "swedish"]

    for text in texts_names:
        path = "texts/" + text + ".txt"
        find_out_language(path, languages, language_names_long)

''' MAIN
'''
if __name__ == "__main__":
    language_names = ["bos", "cro", "czc", "dns", "eng", "frn", "ger", 
                 "gln", "grk", "itn", "nrn", "pol", "por", "rum",
                 "rus", "sco", "slo", "slv", "spn", "swd", "ukr"]
    
    language_names_long = ["bosnian", "croatian", "czech", "danish", "english", "french", "german", 
                 "galician", "greek", "italian", "dutch", "polish", "portuguese", "romanian",
                 "russian", "scottish", "slovak", "slovenian", "spanish", "swedish", "ukrainian"]

    languages = []
    # build vectors for each language and save them to list
    for language in language_names:
        path = "languages/" + language + ".txt"
        languages.append(buildVector(path))

    # first and second part of assignment
    k_medoids_clustering(languages, language_names_long)

    # third part of assignment; comment function above and uncomment this one to see the results
    #finding_out_language(languages)
    