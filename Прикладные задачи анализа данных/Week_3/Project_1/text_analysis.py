import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
# pipeline = Pipeline([
#     ('cove', CountVectorizer()),
#     ('lore', LogisticRegression())])

results = cross_val_score(pipeline, scoring='f1', cv=10, X=messages, y=labels)
print(results)
print(results.mean())
answer5 = round(results.mean(),1)
print('answer5 = ', answer5)
my_write_answer(answer5, 5)

new_data = pd.read_csv('SMS_for_test.txt', header=None)
print(new_data)
print(new_data)
new_messages = list(new_data[0])
# print(new_messages)

pipeline.fit(messages, labels)
new_labels = pipeline.predict(new_messages)
# preds = clf.predict(test_X)
print(new_labels)
answer6 = new_labels

my_write_answer(answer6, 6)

# clf.fit(test_X)
# test_y = clf.predict_proba(test_X)
# print(test_y)