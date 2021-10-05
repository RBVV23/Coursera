import numpy as np
import pandas as pd
import artm
from matplotlib import pyplot as plt
import seaborn
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import random
import matplotlib.cm as cm
from IPython.core.display import display, HTML

seaborn.set_style("whitegrid", {'axes.grid' : False})

batch_vectorizer = artm.BatchVectorizer(data_path="lectures.txt",
                                        data_format="vowpal_wabbit",
                                        target_folder="lectures_batches", batch_size=100)

T = 30
topic_names=["sbj"+str(i) for i in range(T-1)]+["bcg"]
model_artm = artm.ARTM(num_topics=T, topic_names=topic_names, class_ids={"text":1, "author":1},
                       num_document_passes=1, reuse_theta=True, cache_theta=True, seed=-1)

dictionary = artm.Dictionary('dictionary')
dictionary.gather(batch_vectorizer.data_path)

model_artm.seed=1
model_artm.initialize(dictionary=dictionary)

model_artm.scores.add(artm.TopTokensScore(name='TTTScore', class_id="text"))
model_artm.scores.add(artm.TopTokensScore(name='TTAScore', class_id="author"))

model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi',
                                                            tau=1e5, dictionary=dictionary,
                                                            class_ids='text', topic_names='bcg'))
model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi2',
                                                            tau=-1e5, dictionary=dictionary,
                                                            class_ids='text',
                                                            topic_names=topic_names[:-1]))

model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)

tops_words = []
for topic_name in model_artm.topic_names:
    print(topic_name + ': '),
    tokens = model_artm.score_tracker["TTTScore"].last_tokens
    line = ''
    for word in tokens[topic_name]:
        line += word + ' '
    print('\t', line)
    tops_words.append(line)

print()

for topic_name in model_artm.topic_names:
    print(topic_name + ': '),
    tokens = model_artm.score_tracker["TTAScore"].last_tokens
    line = ''
    for word in tokens[topic_name]:
        line += word + ' '
    print('\t', line)

sbj_topic_labels = [
    'Физика элемантарных частиц', #0
    'Химия полимеров', #1
    'Политика Средневековья', #2
    'Высшее образование', #3
    'Нейробиология', #4
    'Греческая мифология', #5
    'Русская литература', #6
    'Расовая теория', #7
    'Римское право', #8
    'Медицина', #9
    'Клеточная биология', #10
    'Астрономия', #11
    'История России', #12
    'Обществознание*', #13
    'Строение Земли', #14
    'Экономика', #15
    'Социология', #16
    'Искусственный интеллект', #17
    'История Древнего мира', #18
    'Генетика', #19
    'Квантовая физика', #20
    'Фмлософия', #21
    'История Франции', #22
    'История СССР', #23
    'Христианство и письменность', #24
    'Линейная алгебра', #25
    'Финансы и феминизм', #26
    'Теория Большого взрыва', #27
    'Кинематограф', #28
]
topic_labels = sbj_topic_labels + ["Фоновая тема"]
print('topic_labels:')
print(topic_labels)

model_artm.theta_columns_naming = "title"
theta = model_artm.get_theta()
phi_a = model_artm.get_phi(class_ids='author')
print('theta.shape: ', theta.shape)
print('phi_a.shape:', phi_a.shape)

plt.figure(figsize=(20,10))
seaborn.heatmap(theta[theta.columns[0:100]])
plt.show()

pt = np.sum(theta, axis=1)/np.sum(np.sum(theta, axis=1), axis=0)
pt.sort_values()
df = pd.DataFrame(pt)
df['topic_labels'] = topic_labels
df['pt']=df[0]
df.drop([0], axis=1, inplace=True)
df=df.sort_values('pt')[:-2]

print('5 наиболее популярных тем:')
print(df['topic_labels'][-5:])
print('3 наименее популярные темы:')
print(df['topic_labels'][:3])

plt.figure(figsize=(20,10))
seaborn.heatmap(phi_a, yticklabels=False)
plt.show()

df = pd.DataFrame(phi_a)
authors = phi_a.index
best_authors = []

Ns = []
for i, author in enumerate(authors):
    counter = 0
    for sbj in topic_names:
        if phi_a[sbj][i] > 0.01:
            counter += 1
    Ns.append(counter)
    if counter >= 3:
        best_authors.append(author)

df.drop(columns=[0, 'N'], axis=0, inplace=True)
df['Significance']=Ns
print('best_authors:')
print(best_authors)

pd.set_option('display.max_columns', 30)

matr=phi_a*pt
vec = np.sum(matr.values, axis=1)
vec = vec.reshape(len(vec),1)
matrix = matr/vec

dists = pairwise_distances(matrix, metric='cosine')
mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
data_2d_mds = mds.fit_transform(dists)

fig = plt.figure(figsize=(15,10))
plt.grid(True)
plt.scatter(data_2d_mds[:, 0], data_2d_mds[:, 1])
plt.savefig('2d_mds_visualization-2')

pd.set_option('display.max_rows', 50)
m = np.array([[1,2,5],
            [7,4,5],
            [4,1,6]])

df=matrix.iloc[:10]
np.argmax(df.iloc[0])

main_tops = []
for i in range(matrix.shape[0]):
    main_tops.append(np.argmax(matrix.iloc[i]))



deltas = np.zeros((T,))
colors = cm.rainbow(np.linspace(0, 1, T))
fig = plt.figure(figsize=(30,20))
plt.grid(True)
for i, p in enumerate(data_2d_mds):
    plt.scatter(p[0], p[1], color=colors[main_tops[i]], s=100, alpha=0.5)
    deltas[main_tops[i]] +=1
    d = deltas[main_tops[i]]
    delta1=[-0.15, 0.01]
    delta2 = -0.03+deltas[main_tops[i]]*0.03
    plt.annotate(authors[i], p+[delta1[int(d%2)],delta2])
plt.savefig('my_new_map.pdf')

vec = np.sum(theta.values, axis=1)
vec = vec.reshape(len(vec),1)
matrix = theta/vec

tops_links = []

for i in range(matrix.shape[0]):
    top_links = matrix.iloc[i].sort_values(ascending=False).index[:10]
    tops_links.append(top_links)

display(HTML(u"<h1>Мой навигатор Постауки</h1>"))
for t in range(T):
    display(HTML(u"<h3>{}</h3>".format(topic_labels[t])))
    display(HTML(u"<h4>{}</h4>".format(tops_words[t])))
    for link in tops_links[t]:
        display(HTML(u'<a href={}>{}</a>'.format(link,link)))