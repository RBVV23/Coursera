from matplotlib import pyplot as plt
import artm

# batch_vectorizer = artm.BatchVectorizer(data_path="school.txt", data_format="vowpal_wabbit", target_folder="school_batches",
#                                        batch_size=100) - необходимо только для первого запуска

batch_vectorizer = artm.BatchVectorizer(data_path='school_batches', data_format='batches')

T = 10
model_artm = artm.ARTM(num_topics=T, topic_names=["sbj"+str(i) for i in range(T)], class_ids={"text":1},
                       num_document_passes=1, reuse_theta=True, cache_theta=True, seed=-1)

dictionary = artm.Dictionary('dictionary')
dictionary.gather(batch_vectorizer.data_path)

model_artm.scores.add(artm.PerplexityScore(name='PerplexityScore',
                                           dictionary='dictionary', ))
model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore', class_id="text"))
model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
model_artm.scores.add(artm.TopTokensScore(name="top_words", num_tokens=15, class_id="text"))

model_artm.initialize('dictionary')

model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=40)
plt.plot(model_artm.score_tracker["PerplexityScore"].value)
plt.show()

for topic_name in model_artm.topic_names:
    print(topic_name + ': '),
    tokens = model_artm.score_tracker["top_words"].last_tokens
    line = ''
    for word in tokens[topic_name]:
        line += word + ' '
    print('\t', line)

print('model_artm.score_tracker["SparsityPhiScore"].last_value = ',
      model_artm.score_tracker["SparsityPhiScore"].last_value)
print('model_artm.score_tracker["SparsityThetaScore"].last_value = ',
      model_artm.score_tracker["SparsityThetaScore"].last_value)

# model_artm.save("my_model")
# model_artm.load("my_model")

phi = model_artm.get_phi()
print('Матрица phi')
print(phi)

theta = model_artm.get_theta()
print('Матрица theta')
print(theta)

model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi',
                                                            tau=-1, dictionary=dictionary))
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=40)

print('После регуляризации с параметром: ', model_artm.regularizers['SparsePhi'].tau)
for topic_name in model_artm.topic_names:
    try:
        print(topic_name + ': '),
        tokens = model_artm.score_tracker["top_words"].last_tokens
        line = ''
        for word in tokens[topic_name]:
            line += word + ' '
        print('\t', line)
    except:
        print('deleted')

print('model_artm.score_tracker["SparsityPhiScore"].last_value = ',
      model_artm.score_tracker["SparsityPhiScore"].last_value)
print('model_artm.score_tracker["SparsityThetaScore"].last_value = ',
      model_artm.score_tracker["SparsityThetaScore"].last_value)

model_artm.regularizers['SparsePhi'].tau = -100
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=40)

print('После регуляризации с параметром: ', model_artm.regularizers['SparsePhi'].tau)
for topic_name in model_artm.topic_names:
    try:
        print(topic_name + ': '),
        tokens = model_artm.score_tracker["top_words"].last_tokens
        line = ''
        for word in tokens[topic_name]:
            line += word + ' '
        print('\t', line)
    except:
        print('deleted')

print('model_artm.score_tracker["SparsityPhiScore"].last_value = ',
      model_artm.score_tracker["SparsityPhiScore"].last_value)
print('model_artm.score_tracker["SparsityThetaScore"].last_value = ',
      model_artm.score_tracker["SparsityThetaScore"].last_value)

model_artm.regularizers['SparsePhi'].tau = -5e4
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=40)

print('После регуляризации с параметром: ', model_artm.regularizers['SparsePhi'].tau)
for topic_name in model_artm.topic_names:
    try:
        print(topic_name + ': '),
        tokens = model_artm.score_tracker["top_words"].last_tokens
        line = ''
        for word in tokens[topic_name]:
            line += word + ' '
        print('\t', line)
    except:
        print('deleted')

print('model_artm.score_tracker["SparsityPhiScore"].last_value = ',
      model_artm.score_tracker["SparsityPhiScore"].last_value)
print('model_artm.score_tracker["SparsityThetaScore"].last_value = ',
      model_artm.score_tracker["SparsityThetaScore"].last_value)

# model_artm.save("my_model") # не будем сохранять не понравившуюся мне модель с регуляризатором

# theta_test = model_artm.transform(batch_vectorizer) - для новых батчей
