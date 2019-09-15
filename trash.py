from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
from fbHash import RollingHash
from time import time
import pandas as pd

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

        def analyser(doc):
            rh = RollingHash(doc,7)
            tokens = []
            for _ in range(7, len(doc) + 1):
                tokens.append(rh.digest())
                rh.update()
            return(tokens)
    
        return(analyser)


if __name__ == "__main__":
    t0 = time()
    # print(list(yield_filepaths('data','xlsx')))
    fpaths = glob.glob(f"**/*.xlsx",recursive=True)
    # vectorizer = CountVectorizer()
    # vectorizer = HashingVectorizer()
    vectorizer = CustomVectorizer()
    # X = vectorizer.fit_transform(yield_filepaths('data','xlsx'))
    X = vectorizer.fit_transform(yield_bytes(fpaths))
    print("vectorized in %fs" % (time() - t0))
    sims_basic = cosine_similarity(X)
    # print(X)
    print()
    print("Basic")
    print(sims_basic)
    print()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    print("transformed in %fs" % (time() - t0))
    # print(vectorizer.get_feature_names())
    # print(tfidf.toarray())
    sims_tfidf = cosine_similarity(tfidf)
    print("Tf-Idf")
    print(sims_tfidf)
    print("cosined in %fs" % (time() - t0))