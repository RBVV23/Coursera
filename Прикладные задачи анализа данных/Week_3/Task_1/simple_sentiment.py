from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from nltk.corpus import movie_reviews
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import FeatureUnion


def text_classifier(vectorizer, transformer, classifier):
    return Pipeline(
            [("vectorizer", vectorizer),
            ("transformer", transformer),
            ("classifier", classifier)]
        )

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

print(negids[:5])

negfeats = [" ".join(movie_reviews.words(fileids=[f])) for f in negids]
posfeats = [" ".join(movie_reviews.words(fileids=[f])) for f in posids]

texts = negfeats + posfeats
labels = [0] * len(negfeats) + [1] * len(posfeats)

print(texts[0])

for clf in [LogisticRegression, LinearSVC, SGDClassifier]:
    print(clf)
    print(cross_val_score(text_classifier(CountVectorizer(), TfidfTransformer(), clf(max_iter=1000)), texts, labels).mean())
    print("\n")

clf_pipeline = Pipeline(
            [("vectorizer", TfidfVectorizer()),
            ("classifier", LinearSVC())]
        )


clf_pipeline.fit(texts, labels)

print(clf_pipeline)

print(clf_pipeline.predict(["Amazing film! I will advice it to all my friends. Genious",
                           "Awful film! The man who advised me to watch it is really crazy idiot."]))


v = CountVectorizer()
mx = v.fit_transform(texts)
mf = TruncatedSVD(10)
u = mf.fit_transform(mx)

for transform in [TruncatedSVD, NMF]:
    print(transform)
    print(cross_val_score(text_classifier(CountVectorizer(), transform(n_components=10), LinearSVC()), texts, labels).mean())
    print("\n")

print(cross_val_score(text_classifier(TfidfVectorizer(), TruncatedSVD(n_components=1000), LinearSVC()),
                      texts,
                      labels
                     ).mean())

print(cross_val_score(
    Pipeline([
            ("vectorizer", CountVectorizer()),
            ("transformer", TruncatedSVD(100)),
            ("classifier", RandomForestClassifier(100))
        ]),
    texts,
    labels
    ))

print(cross_val_score(text_classifier(CountVectorizer(), TruncatedSVD(n_components=1000), RandomForestClassifier(1000)),
                      texts,
                      labels
                     ).mean())

print(cross_val_score(text_classifier(TfidfVectorizer(), TruncatedSVD(n_components=1000), RandomForestClassifier(1000)),
                      texts,
                      labels
                     ).mean())

estimators = [('tfidf', TfidfTransformer()), ('svd', TruncatedSVD(1))]
combined = FeatureUnion(estimators)

print(cross_val_score(
    Pipeline([
            ("vectorizer", CountVectorizer()),
            ("transformer", combined),
            ("classifier", LinearSVC())
        ]),
    texts,
    labels
    ))