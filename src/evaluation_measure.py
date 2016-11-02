import sys, math
from io import open
from pca_reader import PCAprojector
from sklearn.neighbors import NearestNeighbors
import sklearn

#code from https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/rouge/rouge.py
def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

#code from https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/rouge/rouge.py
def calc_rouge(candidate, refs, beta = 1.2):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: float (ROUGE-L score for the candidate evaluated against references)
        """
        assert(len(candidate)==1)
        assert(len(refs)>0)
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs/float(len(token_c)))
            rec.append(lcs/float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if(prec_max!=0 and rec_max !=0):
            score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + beta**2*prec_max)
        else:
            score = 0.0
        return score

def compute_dcg(batcher, input_sentente, ranks, k):
    dcg = 0.
    assert (len(ranks) >= k)
    for pos,ranki_id in enumerate(ranks[:k]):
        i = pos+1
        reference_sentences = batcher.get_captions_txt(ranki_id)
        rel_i = calc_rouge([input_sentente], reference_sentences) # compute relevance as the Rogue
        dcg += (math.pow(2., rel_i) - 1.) / math.log(i + 1., 2.)
    return dcg


def compute_tdcg(batcher, input_sentence, ranks, distances, k):
    tdcg = 0.0
    assert (len(ranks) >= k)
    last_distance = distances[0]
    tie_size = 0
    gain = 0.0
    for pos, ranki_id in enumerate(ranks[:k]):

        distance = distances[pos]
        if distance == last_distance:
            tie_size += 1
        else:
            i = pos + 1
            for j in range(i - tie_size, i):
                tdcg += (math.pow(2.0, gain / tie_size) - 1.0) / math.log(j + 1.0, 2.0)
            last_distance = distance
            tie_size = 1
            gain = 0.0
        reference_sentences = batcher.get_captions_txt(ranki_id)
        gain += calc_rouge([input_sentence], reference_sentences)  # compute relevance as the Rogue
    for j in range(k - tie_size + 1, k + 1):
        tdcg += (math.pow(2.0, gain / tie_size) - 1.0) / math.log(j + 1.0, 2.0)

    return tdcg

#method should be one among 'auto', 'pca', 'cosine'
def evaluation(test_batches, visual_ids, visual_vectors,
               predictions, test_img_ids, test_cap_id,
               predictions_file,
               method='auto',
               mean_file=None, eigen_file=None, pca_num_eig=256, test_loss=None, find_nearest = 25, save_predictions=True):

    if not method in ['pca','cosine']:
        print("Error: method should be one among 'pca','cosine' [Abort]")
        sys.exit()

    proc_predictions = predictions
    nbrs = None

    if method == 'pca':
        print('Normalizing visual features...')
        sklearn.preprocessing.normalize(visual_vectors, norm='l2', axis=1, copy=False)
        proc_predictions = sklearn.preprocessing.normalize(predictions, norm='l2', axis=1, copy=True)

        print('Projecting with PCA')
        pca = PCAprojector(mean_file, eigen_file, visual_vectors.shape[1], num_eig=pca_num_eig)
        visual_vectors = pca.project(visual_vectors)
        proc_predictions = pca.project(proc_predictions)

        nbrs = NearestNeighbors(n_neighbors=find_nearest, n_jobs=-1).fit(visual_vectors)
    else:
        #nbrs = NearestNeighbors(n_neighbors=find_nearest, n_jobs=-1, algorithm='brute', metric='cosine').fit(visual_vectors)
        print('Normalizing visual features...')
        sklearn.preprocessing.normalize(visual_vectors, norm='l2', axis=1, copy=False)
        proc_predictions = sklearn.preprocessing.normalize(predictions, norm='l2', axis=1, copy=True)

        nbrs = NearestNeighbors(n_neighbors=find_nearest, n_jobs=8).fit(visual_vectors)

    print('Getting nearest neighbors...')
    _, indices = nbrs.kneighbors(proc_predictions)

    print('Getting DCG_rouge...')
    dcg_rouge_ave = 0
    tests_processed = 0
    with open(predictions_file, 'w', encoding="utf-8", buffering=100000000) as vis:
      for i in xrange(len(predictions)):
          pred_str = (' '.join(('%.3f' % x) for x in predictions[i])).replace(" 0.000", " 0") if save_predictions else ''
          img_id = test_img_ids[i]
          cap_id = test_cap_id[i]
          cap_txt = test_batches.get_caption_txt(img_id,cap_id)
          nneigbours_ids = visual_ids[indices[i]]
          nneigbours_ids_str = ' '.join([("%d"%x) for x in nneigbours_ids]) if save_predictions else ''
          dcg_rouge = compute_dcg(batcher=test_batches, input_sentente = cap_txt, ranks = nneigbours_ids, k=find_nearest)
          dcg_rouge_ave += dcg_rouge
          if save_predictions:
            vis.write("%s\t%d\t%s\t%s\t%s\t%0.4f\n" % (img_id, cap_id, cap_txt, pred_str, nneigbours_ids_str, dcg_rouge))
          tests_processed += 1
          if tests_processed % 1000 == 0:
              print('Processed %d predictions. DCGAve=%f' % (tests_processed, dcg_rouge_ave / tests_processed))
      dcg_rouge_ave /= tests_processed

      vis.write(u'Test completed: %s DCGrouge=%.4f\n' % (test_loss, dcg_rouge_ave))
      print('Test completed: %s DCGrouge=%.4f' % (test_loss, dcg_rouge_ave))

# def evaluationCosine(test_batches, visual_ids, visual_vectors,
#                      predictions, test_img_ids, test_cap_id,
#                      predictions_file,
#                      test_loss, find_nearest=25, save_predictions=True):
#
#     print('Getting nearest neighbors...')
#     nbrs = NearestNeighbors(n_neighbors=find_nearest, n_jobs=-1, algorithm='brute', metric='cosine').fit(visual_vectors)
#     _, indices = nbrs.kneighbors(predictions)
#
#     print('Getting DCG...')
#     dcg_rouge_ave = 0
#     tests_processed = 0
#     with open(predictions_file, 'w', encoding="utf-8", buffering=100000000) as vis:
#         for i in xrange(len(predictions)):
#             pred_str = (' '.join(('%.3f' % x) for x in predictions[i])).replace(" 0.000"," 0") if save_predictions else ''
#             img_id = test_img_ids[i]
#             cap_id = test_cap_id[i]
#             cap_txt = test_batches.get_caption_txt(img_id, cap_id)
#             nneigbours_ids = visual_ids[indices[i]]
#             nneigbours_ids_str = ' '.join([("%d" % x) for x in nneigbours_ids]) if save_predictions else ''
#             dcg_rouge = compute_dcg(batcher=test_batches, input_sentente=cap_txt, ranks=nneigbours_ids, k=find_nearest)
#             dcg_rouge_ave += dcg_rouge
#             if save_predictions:
#                 vis.write(
#                     "%s\t%d\t%s\t%s\t%s\t%0.4f\n" % (img_id, cap_id, cap_txt, pred_str, nneigbours_ids_str, dcg_rouge))
#             tests_processed += 1
#             if tests_processed % 1000 == 0:
#                 print('Processed %d predictions. DCGAve=%f' % (tests_processed, dcg_rouge_ave/tests_processed))
#         dcg_rouge_ave /= tests_processed
#
#         vis.write(u'Test completed: %s DCGrouge=%.4f\n' % (test_loss, dcg_rouge_ave))
#         print('Test completed: %s DCGrouge=%.4f' % (test_loss, dcg_rouge_ave))

# def evaluationCosine(test_batches, visual_ids, visual_vectors,
#                predictions, test_img_ids, test_cap_id,
#                predictions_file,
#                test_loss, find_nearest = 25, save_predictions=True):
#
#     print('Normalizing visual features...')
#     nbrs = NearestNeighbors(n_neighbors=find_nearest, n_jobs=-1, algorithm='brute', metric='cosine').fit(visual_vectors)
#
#     split_size = 5000
#     n_splits   = len(predictions) / split_size
#     if len(predictions) % split_size > 0: n_splits+=1
#
#     print('Getting DCG_rouge...')
#     dcg_rouge_ave = 0
#     tests_processed = 0
#     with open(predictions_file, 'w', encoding="utf-8", buffering=100000000) as vis:
#         for split in xrange(n_splits):
#             offset = split * split_size
#             pred_batch = predictions[offset:offset+split_size]
#             print('Getting nearest neighbors...')
#             _, indices = nbrs.kneighbors(pred_batch)
#             for i in xrange(len(pred_batch)):
#                 pred_str = (' '.join(('%.3f' % x) for x in pred_batch[i])).replace(" 0.000"," 0") if save_predictions else ''
#                 img_id = test_img_ids[offset+i]
#                 cap_id = test_cap_id[offset+i]
#                 cap_txt = test_batches.get_caption_txt(img_id, cap_id)
#                 nneigbours_ids = visual_ids[indices[i]]
#                 nneigbours_ids_str = ' '.join([("%d" % x) for x in nneigbours_ids]) if save_predictions else ''
#                 dcg_rouge = compute_dcg(batcher=test_batches, input_sentente=cap_txt, ranks=nneigbours_ids,
#                                         k=find_nearest)
#                 dcg_rouge_ave += dcg_rouge
#                 if save_predictions:
#                     vis.write("%s\t%d\t%s\t%s\t%s\t%0.4f\n" % (
#                     img_id, cap_id, cap_txt, pred_str, nneigbours_ids_str, dcg_rouge))
#                 tests_processed += 1
#             print('Processed %d predictions' % (tests_processed))
#         dcg_rouge_ave /= tests_processed
#
#         vis.write(u'Test completed: %s DCGrouge=%.4f\n' % (test_loss, dcg_rouge_ave))
#         print('Test completed: %s DCGrouge=%.4f' % (test_loss, dcg_rouge_ave))
