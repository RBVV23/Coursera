import json
from gensim import corpora, models
import numpy as np
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas
import seaborn
from matplotlib import pyplot as plt

def save_answers1(c_salt, c_sugar, c_water, c_mushrooms, c_chicken, c_eggs):
    with open("cooking_LDA_pa_task1.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [c_salt, c_sugar, c_water, c_mushrooms, c_chicken, c_eggs]]))
def save_answers2(dict_size_before, dict_size_after, corpus_size_before, corpus_size_after):
    with open("cooking_LDA_pa_task2.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [dict_size_before, dict_size_after, corpus_size_before, corpus_size_after]]))
def save_answers3(coherence, coherence2):
    with open("cooking_LDA_pa_task3.txt", "w") as fout:
        fout.write(" ".join(["%3f"%el for el in [coherence, coherence2]]))
def save_answers4(count_model2, count_model3):
    with open("cooking_LDA_pa_task4.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [count_model2, count_model3]]))
def save_answers5(accuracy):
    with open("cooking_LDA_pa_task5.txt", "w") as fout:
        fout.write(str(accuracy))

def generate_recipe(model, num_ingredients):
    theta = np.random.dirichlet(model.alpha)
    for i in range(num_ingredients):
        t = np.random.choice(np.arange(model.num_topics), p=theta)
        topic = model.show_topic(t, topn=model.num_terms)
        topic_distr = [x[1] for x in topic]
        terms = [x[0] for x in topic]
        w = np.random.choice(terms, p=topic_distr)
        print(w)
def compute_topic_cuisine_matrix(model, corpus, recipes):
    targets = list(set([recipe["cuisine"] for recipe in recipes]))
    tc_matrix = pandas.DataFrame(data=np.zeros((model.num_topics, len(targets))), columns=targets)
    for recipe, bow in zip(recipes, corpus):
        recipe_topic = model.get_document_topics(bow)
        for t, prob in recipe_topic:
            tc_matrix[recipe["cuisine"]][t] += prob
    target_sums = pandas.DataFrame(data=np.zeros((1, len(targets))), columns=targets)
    for recipe in recipes:
        target_sums[recipe["cuisine"]] += 1
    return pandas.DataFrame(tc_matrix.values/target_sums.values, columns=tc_matrix.columns)
def plot_matrix(tc_matrix):
    plt.figure(figsize=(10, 10))
    seaborn.heatmap(tc_matrix, square=True)

def my_mean_coherence(my_lda, my_corpus):
    sum = 0
    var = my_lda.top_topics(my_corpus)
    for i in range(len(var)):
        sum += var[i][-1]
    my_coherence = sum/len(var)
    return my_coherence
def my_sum(my_lda, my_corpus):
    sum = 0
    for line in my_lda.get_document_topics(my_corpus, minimum_probability=0.01):
        sum += len(line)
    return sum
def my_y_maker(recipes, cuisines):
    y = []
    for rec in recipes:
        targ = cuisines.index(rec['cuisine'])
        y.append(targ)
    return y

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
save_answers2(dict_size_before, dict_size_after, corpus_size_before, corpus_size_after)

np.random.seed(76543)
lda2 = models.ldamodel.LdaModel(corpus2, num_topics=40, id2word=dictionary2, passes=5)

coherence = my_mean_coherence(lda, corpus)
coherence2 = my_mean_coherence(lda2, corpus2)

print('coherence = ', coherence)
print('coherence2 = ', coherence2)
save_answers3(coherence, coherence2)

print('lda2.get_document_topics(corpus2)[0]:')
print(lda2.get_document_topics(corpus2)[0])

print('lda2.alpha:')
print(lda2.alpha)

np.random.seed(76543)
lda3 = models.ldamodel.LdaModel(corpus2, num_topics=40, id2word=dictionary2, passes=5, alpha=1)
print('lda3.get_document_topics(corpus2)[0]:')
print(lda3.get_document_topics(corpus2)[0])

count_model2 = my_sum(lda2, corpus2)
count_model3 = my_sum(lda3, corpus2)
print('count_model2 = ', count_model2)
print('count_model3 = ', count_model3)
save_answers4(count_model2, count_model3)

cuisines = []
for rec in recipes:
    if not (rec['cuisine'] in cuisines):
        cuisines.append(rec['cuisine'])

targets = my_y_maker(recipes, cuisines)

X = np.zeros((len(lda2.get_document_topics(corpus2)), lda2.num_topics))

for i,line in enumerate(lda2.get_document_topics(corpus2)):
    for top in line:
        X[i][top[0]] = top[1]

estimator = RandomForestClassifier(n_estimators=100)
result = cross_val_score(estimator, X, targets, cv=3, scoring='accuracy')

print(result)
accuracy=np.mean(result, axis=0)
print('accuracy = ', accuracy)
save_answers5(accuracy)

print('generate_recipe: ')
np.random.seed(0)
generate_recipe(lda2, 5)

tc_matrix = compute_topic_cuisine_matrix(lda2, corpus2, recipes)
plot_matrix(tc_matrix)
