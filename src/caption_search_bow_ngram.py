import argparse
import sys
from collections import defaultdict

import time
from sklearn.preprocessing import normalize

from evaluation_measure import compute_tdcg
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

stopwords = ['of', 'the', 'a', 'an']


class MyBatcher():
    def __init__(self, ids, labels, docs):
        self._data = defaultdict(list)
        for id_, doc in zip(ids, docs):
            self._data[id_].append(doc)

    def get_captions_txt(self, id_):
        return self._data[id_]


def read_data(filename, count=0, filter_ids=None, filter_label=None):
    ids = list()
    labels = list()
    docs = list()
    i = 0
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            id_, label, text = line.split('\t')[:3]
            if (not filter_ids or id_ in filter_ids) and (not filter_label or label in filter_label):
                ids.append(id_)
                labels.append(label)
                docs.append(text)
                i += 1
                if i == count:
                    break
    return ids, labels, docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--queries', type=str, default='Test_val2014.sentences.txt')
    # parser.add_argument('-c', '--corpus', type=str, default='Test_val2014_generated_captions_neuraltalk_v2.txt')
    # parser.add_argument('-c', '--corpus', type=str, default='val2014_generated_captions_show_and_tell_iter_1M.txt')
    parser.add_argument('-c', '--corpus', type=str, default='val2014_generated_captions_show_and_tell_iter_2M.txt')
    args = parser.parse_args(sys.argv[1:])

    # count = 400
    # query_ids, query_labels, queries = read_data(args.queries, count)
    # ids, labels, docs = read_data(args.corpus, count, set(query_ids))

    # query_ids, query_labels, queries = read_data(args.queries)
    # ids, labels, docs = read_data(args.corpus)

    # query_ids, query_labels, queries = read_data(args.queries, count)
    # ids, labels, docs = read_data(args.corpus, count, set(query_ids))

    query_ids, query_labels, queries = read_data(args.queries)
    ids, labels, docs = read_data(args.corpus, filter_ids=set(query_ids), filter_label=['0'])

    print('queries', set(query_labels), len(query_ids))
    print('corpus', set(labels), len(ids))

    batcher = MyBatcher(query_ids, query_labels, queries)

    # vectorizer = CountVectorizer(min_df=1, stop_words=stopwords)
    # vectorizer = CountVectorizer(min_df=5, stop_words=stopwords)
    vectorizer = CountVectorizer(min_df=5, stop_words=stopwords, ngram_range=(3, 4), analyzer='char')
    tfidfer = TfidfTransformer(norm='l2')

    tfidf_corpus = tfidfer.fit_transform(vectorizer.fit_transform(docs))

    tfidf_queries = tfidfer.transform(vectorizer.transform(queries))

    tfidf_corpus = normalize(tfidf_corpus, norm='l2', axis=1, copy=True)
    tfidf_queries = normalize(tfidf_queries, norm='l2', axis=1, copy=True)

    k = min(25, tfidf_corpus.shape[0])

    print(tfidf_corpus.shape)
    print(tfidf_queries.shape)

    start = time.time()

    atdcg = 0.0
    block_size = 1000
    blocks = 0
    block = tfidf_queries[block_size * blocks:block_size * (blocks + 1)]
    while blocks < 1:
        blocks += 1
        if block_size * blocks <= tfidf_queries.shape[0]:
            block = tfidf_queries[block_size * blocks:min(block_size * (blocks + 1), tfidf_queries.shape[0])]
            block_distances = euclidean_distances(block, tfidf_corpus)

            for query_id, query, query_distances in zip(
                    query_ids[block_size * blocks:min(block_size * (blocks + 1), tfidf_queries.shape[0])],
                    queries[block_size * blocks:min(block_size * (blocks + 1), tfidf_queries.shape[0])],
                    block_distances):
                ranks = [id_ for (distance, id_) in sorted(zip(query_distances, ids))]
                value = compute_tdcg(batcher, query, ranks, sorted(query_distances), k)
                atdcg += value
        else:
            break

    done = time.time()
    elapsed = done - start
    print(elapsed)

    atdcg /= len(queries)

    print('eucl ATDCG', atdcg)
