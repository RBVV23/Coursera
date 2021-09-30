import json
from gensim import corpora, models
import numpy as np
import copy

def save_answers1(c_salt, c_sugar, c_water, c_mushrooms, c_chicken, c_eggs):
    with open("cooking_LDA_pa_task1.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [c_salt, c_sugar, c_water, c_mushrooms, c_chicken, c_eggs]]))


with open("recipes.json") as f:
    recipes = json.load(f)
print('recipes[0]:')
print(recipes[0])

texts = [recipe["ingredients"] for recipe in recipes]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print('texts[0] и corpus[0]:')
print(texts[0])
print(corpus[0])

np.random.seed(76543)
lda = models.ldamodel.LdaModel(corpus, num_topics=40, id2word=dictionary, passes=5)

np.random.seed(76543)
tops = lda.show_topics(formatted=False, num_topics=40, num_words=10)

target_ingr = ["salt", "sugar", "water", "mushrooms", "chicken", "eggs"]
c_ingrs = np.zeros(len(target_ingr))
c_ingrs = list(map(lambda x: int(x), c_ingrs))

for topic in tops:
    for ingr in topic[1]:
        for i in range(len(target_ingr)):
            if ingr[0] == target_ingr[i]:
                c_ingrs[i] += 1

save_answers1(c_ingrs[0], c_ingrs[1], c_ingrs[2], c_ingrs[3], c_ingrs[4], c_ingrs[5])
print('c_ingrs = ', c_ingrs)


dictionary2 = copy.deepcopy(dictionary)
dict_size_before = len(dictionary2)
print('dict_size_before = ', dict_size_before)

top_ingrs = []
top_ingrs_ids = []
for key in dictionary2.dfs:
    if dictionary2.dfs[key] > 4000:
        top_ingrs.append(dictionary2[key])
        top_ingrs_ids.append(key)

print('top_ingrs = ', top_ingrs)
print('top_ingrs_ids = ', top_ingrs_ids)

dictionary2.filter_tokens(top_ingrs_ids)
dict_size_after = len(dictionary2)
print('dict_size_after = ', dict_size_after)

corpus_size_before = 0
for doc in corpus:
    corpus_size_before += len(doc)

print('corpus_size_before = ', corpus_size_before)

corpus2 = [dictionary2.doc2bow(text) for text in texts]

corpus_size_after = 0
for doc in corpus2:
    corpus_size_after += len(doc)

print('corpus_size_after = ', corpus_size_after)