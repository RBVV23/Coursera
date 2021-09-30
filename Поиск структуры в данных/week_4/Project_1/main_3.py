import json
from gensim import corpora, models
import numpy as np



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
