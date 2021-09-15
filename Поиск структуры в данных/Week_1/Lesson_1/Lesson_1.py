from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
import itertools

def validate_with_mappings(preds, target, dataset):
    permutations = itertools.permutations([0,1,2])
    for a,b,c in permutations:
        mapping = {2:a, 1:b, 0:c}
        mapped_preds = [mapping[pred] for pred in preds]
        print('\n',mapping)
        print('Точность: ', sum(mapped_preds == target)/len(target))


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

model = AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='complete')
preds = model.fit_predict(matrix.toarray())

print('Вектор предсказаний: ',preds)
print('Нулевой объект матрицы:')
print(matrix[0])

print('Имена признаков: ')
print(vectorizer.get_feature_names())

print('Рассмотрим первый (в матрице) признак-слово первого объекта [877]: ')
print(vectorizer.get_feature_names()[877])
print('Этот признак действительно содержится в первом тексте:')
print(simple_dataset.data[0])


model = KMeans(n_clusters=3, random_state=1)
preds = model.fit_predict(matrix.toarray())

print('Вектор предсказаний: ',preds)
print('Вектор ответов: ',simple_dataset.target)

mapping = {0:0, 2:1, 1:2}
mapped_preds = [mapping[pred] for pred in preds]
print('Точность: ', sum(mapped_preds == simple_dataset.target)/len(simple_dataset.target))

clf = LogisticRegression()
print('Сравним с логистической регрессией:')
print(cross_val_score(clf, matrix, simple_dataset.target).mean())


dataset = fetch_20newsgroups(subset='train',
                             categories=['comp.sys.mac.hardware',
                                         'comp.os.ms-windows.misc', 'comp.graphics'])
matrix = vectorizer.fit_transform(dataset.data)
model = KMeans(n_clusters=3, random_state=42)
preds = model.fit_predict(matrix.toarray())
print('Вектор предсказаний: ',preds)
print('Вектор ответов: ',dataset.target)

mapping = {0:0, 2:2, 1:1}
mapped_preds = [mapping[pred] for pred in preds]
print('Точность: ', sum(mapped_preds == dataset.target)/len(dataset.target))

print('Сравним с логистической регрессией:')
print(cross_val_score(clf, matrix, dataset.target).mean())

svd = TruncatedSVD(n_components=1000, random_state=123)
features = svd.fit_transform(matrix)
preds = model.fit_predict(features)
print('Вектор предсказаний: ',preds)
print('Вектор ответов: ',dataset.target)

## mapping = {0:2, 1:1, 2:0}
## mapped_preds = [mapping[pred] for pred in preds]
## print('Точность {0:2, 1:1, 2:0}: ', sum(mapped_preds == dataset.target)/len(dataset.target))

## mapping = {0:0, 1:1, 2:2}
## mapped_preds = [mapping[pred] for pred in preds]
## print('Точность {0:0, 1:1, 2:2}: ', sum(mapped_preds == dataset.target)/len(dataset.target))

mapping = {0:2, 1:0, 2:1}
mapped_preds = [mapping[pred] for pred in preds]
print('Точность {0:2, 1:0, 2:1}: ', sum(mapped_preds == dataset.target)/len(dataset.target))

svd = TruncatedSVD(n_components=200, random_state=123)
features = svd.fit_transform(matrix)
preds = model.fit_predict(features)
print('Вектор предсказаний: ',preds)
print('Вектор ответов: ',dataset.target)

validate_with_mappings(preds, dataset.target, dataset)