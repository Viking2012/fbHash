import glob
import os
from time import time

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fbHash import RollingHash


def yield_filepaths(parent_dir, extension='txt'):
    for fpath in glob.glob(f"**/*.{extension}",recursive=True):
        with open(fpath, 'rb') as f:
            yield f.read()

def yield_bytes(fpaths):
    for fpath in fpaths:
        with open(fpath,'rb') as f:
            yield f.read()

class CustomVectorizer(CountVectorizer):
    def build_analyzer(self):

        def analyser(doc, k=7, a=26, mod=None):
            rh = RollingHash(doc, k=k, a=a, mod=mod)
            tokens = []
            for _ in range(7, len(doc) + 1):
                tokens.append(rh.digest())
                rh.update()
            return(tokens)
    
        return(analyser)

def extract_ext(fpath):
    fname, fext = os.path.splitext(fpath)
    return fext

if __name__ == "__main__":
    t0 = time()
    fpaths = []
    fpaths.extend(glob.glob(f"**/*.txt",recursive=True))
    fpaths.extend(glob.glob(f"**/*.xlsx",recursive=True))
    fpaths.extend(glob.glob(f"**/*.py",recursive=True))
    vectorizer = CustomVectorizer()
    X = vectorizer.fit_transform(yield_bytes(fpaths))
    print("vectorized in %fs" % (time() - t0))
    sims_basic = cosine_similarity(X)
    print()
    # print("Basic")
    # print(sims_basic)
    print()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    print("transformed in %fs" % (time() - t0))
    sims_tfidf = cosine_similarity(tfidf)
    # print("Tf-Idf")
    # print(sims_tfidf)
    print("cosined in %fs" % (time() - t0))

    for i,j in np.argwhere(sims_tfidf>0.65):
        if i >= j:
            continue
        print(f"{fpaths[i]} matched with {fpaths[j]} at {sims_tfidf[i][j]:.3%}%")

    labels = [extract_ext(f) for f in fpaths]

    # km = KMeans(n_clusters = len(set(labels)), init='k-means++', max_iter=100, n_init=1)
    km = AgglomerativeClustering(n_clusters = len(set(labels)))
    km.fit(tfidf.toarray())
    print("clustered in %fs" % (time() - t0))

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, km.labels_, sample_size=1000))

    for i in range(len(fpaths)):
        print("{:>45} {:>5} {}".format(
            fpaths[i],
            labels[i],
            km.labels_[i])
        )