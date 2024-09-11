import json
import random

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

data = []
with open('metadata.jsonl', 'r') as file:
    for line in file:
        data.append(json.loads(line))

sentences = []
dictionary = {}
for item in data:
    file_name = item['file_name']
    text = item['text']
    dictionary[file_name] = text[text.find(',') + 2:].split(' - ')
    sentences.append(text)
embeddings = model.encode(sentences)


def random_biome(biome):
    """
    returns a link to a random map of the given biome
    """
    point = random.randint(0, len(dictionary) - 1)
    for i in range(0, len(dictionary)):
        if point + i == len(dictionary):
            point = -i
        if dictionary[list(dictionary.keys())[point + i]][0] == biome:
            return list(dictionary.keys())[point + i]
    return 'nothing was found'


def similar_description(user_description):
    """
    returns an array of three link to maps that best match the given description in descending order of suitability
    """
    user_embedding = model.encode(user_description)
    cos_sim = util.cos_sim(embeddings, user_embedding)
    sim_arr = []
    for i in range(len(cos_sim) - 1):
        sim_arr.append([cos_sim[i], i])
    sim_arr = sorted(sim_arr, key=lambda x: x[0], reverse=True)
    ans = []
    for score, i in sim_arr[0:3]:
        ans.append("img/image_" + str(i + 1).zfill(3) + ".jpg")
    return ans
