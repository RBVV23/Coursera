from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster.hierarchical import AgglomerativeClustering

simple_dataset = fetch_20newsgroups(subset='train',
                                    categories=['comp.sys.mac.hardware', 'soc.religion.christian',
                                                'rec.sport.hockey'])
print('Темы: ', simple_dataset.target_names)

print('Пример объекта (текста):')
# print(simple_dataset.data[0])
print('simple_dataset.target: ', simple_dataset.target)

print('Всего объектов: ', len(simple_dataset.data))

vectorizer = TfidfVectorizer(max_df=500, min_df=10)
matrix = vectorizer.fit_transform(simple_dataset.data)
print('matrix.shape = ', matrix.shape)

