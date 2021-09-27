import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gensim import corpora, models

data = corpora.UciCorpus('docword.xkcd.txt', 'vocab.xkcd.txt')
dictionary = data.create_dictionary()

# ldamodel = models.ldamodel.LdaModel(data, id2word=dictionary, num_topics=5, passes=20,
#                                     alpha=1.5, eta=1.5)
# ldamodel.save('ldamodel_xkcd')

ldamodel = models.ldamodel.LdaModel.load('ldamodel_xkcd')

for t, top_words in ldamodel.print_topics(num_topics=10, num_words=10):
    print('Topic', t, ':', top_words)

perplexity = ldamodel.log_perplexity(list(data))
print(2**(-perplexity))

perp = ldamodel.bound(data)
print(2**(-perp/float(87409)))

# data2 = corpora.UciCorpus('docword2.xkcd.txt', 'vocab2.xkcd.txt')
# ldamodel.update(data2, passes=10) - добавление новых данных при наличии

doc = list(data)[0]
print(ldamodel.get_document_topics(doc))