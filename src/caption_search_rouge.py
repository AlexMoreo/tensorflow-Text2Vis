import argparse
import sys
from collections import defaultdict
from multiprocessing import Pool, Process, Manager
import functools
import time

from evaluation_measure import compute_tdcg, calc_rouge

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


def process_query(query, docs, ids, batcher, k, reverse, q):
    rouges = list()
    for doc in docs:
        rouges.append(calc_rouge([query], [doc]))
    ranks = [id_ for (distance, id_) in sorted(zip(rouges, ids), reverse=reverse)]
    tdcg = compute_tdcg(batcher, query, ranks, sorted(rouges, reverse=reverse), k)
    q.put(tdcg)
    return tdcg


def incremental_average(q):
    count = 0
    sum = 0
    for value in iter(q.get, None):
        count += 1
        sum += value
        print(sum / count, count, flush=True)


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

    query_ids, query_labels, queries = read_data(args.queries)
    ids, labels, docs = read_data(args.corpus, filter_ids=set(query_ids), filter_label=['0'])

    print('queries', set(query_labels), len(query_ids))
    print('corpus', set(labels), len(ids))

    batcher = MyBatcher(query_ids, query_labels, queries)

    k = 25
    den = len(queries)
    processes = 1
    reverse = True

    manager = Manager()
    q = manager.Queue()

    p = Process(target=incremental_average, args=(q,))
    p.start()

    start = time.time()

    pool = Pool(processes=processes)
    ret = pool.map(functools.partial(process_query, docs=docs, ids=ids, batcher=batcher, k=k, reverse=reverse, q=q),
                   queries[:den])
    print('len(ret)', len(ret))
    atdcg = sum(ret)

    atdcg /= den

    done = time.time()
    elapsed = done - start
    print(elapsed)
    print('roug ATDCG', atdcg)

    q.put(None)
