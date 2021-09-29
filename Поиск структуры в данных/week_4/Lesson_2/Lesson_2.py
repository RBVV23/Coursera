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