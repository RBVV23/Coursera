import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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

vectorizer = CountVectorizer(input=messages)
X = vectorizer.transform()
print(X)

