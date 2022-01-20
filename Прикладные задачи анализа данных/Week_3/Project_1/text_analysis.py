import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def my_write_answer(answer, number):
    name = 'answer' + str(number) + '.txt'
    with open(name, 'w') as file:
        file.write(str(answer))

data = pd.read_csv('SMSSpamCollection.txt', sep='\t', header=None)

data.columns = ['content', 'sms']

print(data.head())
print(data.shape)
print(data.describe())

labels = [1 if (x == 'spam') else 0 for x in data.content]
# print(labels)
print(type(labels))

messages = list(data.sms)
# print(messages)
print(type(messages))

# X = CountVectorizer(input=messages)
# print(X)

# vectorizer = CountVectorizer()
X = CountVectorizer().fit_transform(messages)
print(X.shape)
# print(X)

# [0.97297297 0.89855072 0.91428571 0.95774648 0.92957746 0.91304348
#  0.92957746 0.92857143 0.91549296 0.95172414]

pipeline = Pipeline([
    ('Создание_вектора_признаков', CountVectorizer()),
    ('Классификация_логистической_регрессией', LogisticRegression())])

results_unigramms = cross_val_score(pipeline, scoring='f1', cv=10, X=messages, y=labels)
print(results_unigramms)
print(results_unigramms.mean())
answer5 = round(results_unigramms.mean(),1)
print('answer5 = ', answer5)
my_write_answer(answer5, 5)

new_data = pd.read_csv('SMS_for_test.txt', header=None)
print(new_data)
print(new_data)
new_messages = list(new_data[0])
# print(new_messages)

pipeline.fit(messages, labels)
new_labels = pipeline.predict(new_messages)

answer6 = str(new_labels)[1:-1]
print('answer6 = ', answer6)
my_write_answer(answer6, 6)

results = []
params = [(2,2), (3,3), (1,3)]
for trpl in params:
    pipeline_2 = Pipeline([
        ('Создание_вектора_признаков', CountVectorizer(ngram_range=trpl)),
        ('Классификация_логистической_регрессией', LogisticRegression())])

    result = cross_val_score(pipeline_2, scoring='f1', cv=10, X=messages, y=labels).mean()
    results.append(round(result,2))

print(results)

answer7 = str(results)[1:-1]
answer7 = answer7[:4] + answer7[5:10] + answer7[11:16]
print('answer7 = ', answer7)
my_write_answer(answer7, 7)

results = []
params = [(2,2), (3,3), (1,3)]
for trpl in params:
    X = CountVectorizer(ngram_range=trpl).fit_transform(messages)
    clf = MultinomialNB()
    result = cross_val_score(clf, scoring='f1', cv=10, X=X, y=labels).mean()
    results.append(round(result,2))

print(results)

answer8 = str(results)[1:-1]
answer8 = answer8[:4] + answer8[5:10] + answer8[11:16]
print('answer8 = ', answer8)
my_write_answer(answer8, 8)

pipeline_4 = Pipeline([
    ('Создание_вектора_признаков', TfidfVectorizer()),
    ('Классификация_логистической_регрессией', LogisticRegression())])

results = cross_val_score(pipeline_4, scoring='f1', cv=10, X=messages, y=labels)
print(results)
print('results_unigramms.mean() = ', results_unigramms.mean())
print('results.mean() = ', results.mean())

answer9 = 11
print('abs(results.mean() - results_unigramms.mean()) = ', abs(results.mean() - results_unigramms.mean()))

if abs(results.mean() - results_unigramms.mean()) < 0.01:
    answer9 = 0
    print('0')
elif results.mean() > results_unigramms.mean():
    answer9 = 1
    print('1')
else:
    answer9 = -1
    print('-1')

print('answer9 = ', answer9)
my_write_answer(answer9, 9)
