import glob
import os
from pprint import pprint
from random import randrange
from time import time

import numpy as np
import pandas as pd
from sklearn import cluster, metrics
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer,
                                             TfidfTransformer, 
                                             TfidfVectorizer)
from sklearn.metrics.pairwise import cosine_similarity


def yield_filepaths(parent_dir, extension='txt'):
    for fpath in glob.glob(f"**/*.{extension}",recursive=True):
        with open(fpath, 'rb') as f:
            yield f.read()

def yield_bytes(fpaths):
    for fpath in fpaths:
        with open(fpath,'rb') as f:
            yield f.read()
class RollingHash:
    known_64_bit_primes = [
            17586613600806056593,
            10324706610870574883,
            14385965969526276271,
            15700719402893486197,
            13390804203280917121,
            12631952504492069741,
            14687623246052906689,
            18235099962527857067,
            13557970565612484931,
        ]
    def __init__(self, bytestring, k=7, a=26, mod=None):
        self.b  = bytestring
        self.hash = 0
        self.a = a

        if mod is None:
            self.mod = self.large_primes(1)
        else:
            self.mod = mod
		
        for i in range(0, k):
            self.hash = self._finalize_hash(self.hash, 0, self.b[i])

        self.init = 0
        self.end  = k
		
    def update(self):
        if self.end <= len(self.b) - 1:
            old = self.b[self.init]
            new = self.b[self.end]
            self.hash = self._finalize_hash(self.hash, old, new)
            self.init += 1
            self.end  += 1

    def _finalize_hash(self, current, old, new):
        current -= old * self.a
        current += new * self.a
        # current -= old
        # current += new
        current += self.mod
        return current % self.mod
            
    def digest(self):
        return self.hash

    def text(self,encoding='utf-8'):
        return self.b[self.init:self.end].decode(encoding)

    def large_primes(self,idx=None):
        if idx is None:
            idx = randrange(len(self.known_64_bit_primes))
        return self.known_64_bit_primes[idx]


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
    _, fext = os.path.splitext(fpath)
    return fext

def save_csv(outpath, arr, index_list=None, col_list=None):
    df = pd.DataFrame(arr)
    if index_list is not None:
        df.index = index_list
    if col_list is not None:
        df.columns = col_list
    df.to_csv(outpath)

if __name__ == "__main__":
    t0 = time()
    fpaths = []
    fpaths.extend(glob.glob(f"**/*.txt",recursive=True))
    fpaths.extend(glob.glob(f"**/*.xlsx",recursive=True))
    fpaths.extend(glob.glob(f"**/*.py",recursive=True))
    # fpaths.extend(glob.glob("path/to/folder/*.*",recursive=True))
    # fpaths.sort()
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
    label_map = list(set(labels))

    # for f,l in zip(fpaths, labels):
    #     print(f"For {f} got {l}")

    km = cluster.AgglomerativeClustering(n_clusters = len(set(labels)))
    km = cluster.AffinityPropagation()
    km.fit(tfidf.toarray())
    print("clustered in %fs" % (time() - t0))

    print(km.__dict__)

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

    save_csv(
        outpath='fbHash/results/labeled_df.csv', 
        arr=tfidf.toarray(), 
        index_list = fpaths)
    save_csv(
        outpath='fbHash/results/tf_idf_similarity.csv', 
        arr=sims_tfidf
        )
