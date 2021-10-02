import numpy as np
import pandas as pd
import artm
from matplotlib import pyplot as plt
import seaborn
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