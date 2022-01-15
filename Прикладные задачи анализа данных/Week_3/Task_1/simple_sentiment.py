from nltk.corpus import movie_reviews
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

print(negids[:5])

negfeats = [" ".join(movie_reviews.words(fileids=[f])) for f in negids]
posfeats = [" ".join(movie_reviews.words(fileids=[f])) for f in posids]

texts = negfeats + posfeats
labels = [0] * len(negfeats) + [1] * len(posfeats)

print(texts[0])