import sys, os
import random
import numpy as np
from mscoco_captions_reader import MSCocoCaptions
from visual_embeddings_reader import VisualEmbeddingsReader

class Batcher:

    def __init__(self,
                 captions_file,
                 visual_file,
                 buckets_def,
                 batch_size,
                 lemma=False,
                 dowide=True,
                 word2id=None, id2word=None):
        self._captions = MSCocoCaptions(captions_file, word2id=word2id, id2word=id2word, lemma=lemma)
        self._visual = VisualEmbeddingsReader(visual_file)
        self._batch_size = batch_size
        self._build_buckets(buckets_def) # list of bucket sizes
        self.epoch = 0 #automatically increases when all examples have been seen
        self._is_cached_pads = True
        self._dowide=dowide
        self._cache_pads = dict()

    def _get_bucket(self, caption, buckets_def):
        l = len(caption)
        for x in buckets_def:
            if x >= l: return x
        #if no bucket can contain it, returns the biggest
        return max(buckets_def)

    def _build_buckets(self, buckets_def):
        print('Building buckets...')
        self.buckets = dict([(x, []) for x in buckets_def])
        for img_label in self._captions.get_image_ids():
            for cap_pos, cap in enumerate(self._captions.get_captions(img_label)):
                bucket = self._get_bucket(cap, buckets_def)
                self.buckets[bucket].append([img_label, cap_pos])
        print('\t' + ' '.join([('[size=%d, %d]'%(x,len(self.buckets[x]))) for x in buckets_def]))
        self.buckets_def = []
        for bucket_size in buckets_def:
            bucket_length = len(self.buckets[bucket_size])
            if bucket_length < self._batch_size:
                print('Warning: bucket %d contains only %d elements.' % (bucket_size, bucket_length))
            #    del self.buckets[bucket_size]
            #    print('Removing bucket %d, it contains only %d elements [%d required]' % (bucket_size, bucket_length, self._batch_size))
            #else:
            self.buckets_def.append(bucket_size)
        self.buckets_def.sort()
        self._curr_bucket_pos = 0 # current bucket position
        self._offset_bucket = 0  # offset position in the current bucket block

    def current_bucket_elements(self):
        return self.buckets[self.current_bucket_size()]

    def current_bucket_size(self):
        return self.buckets_def[self._curr_bucket_pos]

    def num_buckets(self):
        return len(self.buckets_def)

    def _next_bucket(self):
        self._curr_bucket_pos = (self._curr_bucket_pos + 1) % self.num_buckets()
        self._offset_bucket=0
        random.shuffle(self.current_bucket_elements())
        if self._curr_bucket_pos==0: self.epoch += 1

    def _get_pad(self, img_label, cap_pos, bucket_size):
        if self._is_cached_pads:
            if img_label not in self._cache_pads:
                self._cache_pads[img_label]=dict()
            if cap_pos not in self._cache_pads[img_label]:
                self._cache_pads[img_label][cap_pos]=self._gen_pad(img_label, cap_pos, bucket_size)
            return self._cache_pads[img_label][cap_pos]
        else:
            return self._gen_pad(img_label, cap_pos, bucket_size)

    """
    Applies padding to the caption until bucket_size
    If the caption length is greater than bucket_size, the endding part of the caption is cut
    """
    def _gen_pad(self, img_label, cap_pos, bucket_size):
        caption = self._captions.get_captions(img_label)[cap_pos]
        num_pads = max(bucket_size - len(caption), 0)
        return [self._captions.get_pad()] * num_pads + caption[:bucket_size]

    def wide1hot(self, img_label, cap_pos):
        if not self._dowide: return None
        caption = self._captions.get_captions(img_label)[cap_pos]
        wide1hot = np.zeros(self.vocabulary_size())
        wide1hot[caption] = 1
        return wide1hot

    def next(self):
        caps = self.current_bucket_elements()[self._offset_bucket: self._offset_bucket + self._batch_size]
        img_labels, caps_pos, pads, visuals = [],[],[],[]
        wide1hot = []
        for (img_label, cap_pos) in caps:
            img_labels.append(img_label)
            caps_pos.append(cap_pos)
            wide1hot.append(self.wide1hot(img_label, cap_pos))
            pads += self._get_pad(img_label, cap_pos, self.buckets_def[self._curr_bucket_pos])
            visuals.append(self._visual.get(img_label))
        pads = np.array(pads).reshape(len(caps), -1)
        current_bucket_size = self.current_bucket_size()
        self._offset_bucket += self._batch_size
        if self._offset_bucket >= len(self.current_bucket_elements()):  # check if the bucket has been processed
            self._next_bucket()
        return img_labels, caps_pos, pads, wide1hot, visuals, current_bucket_size

    def vocabulary_size(self):
        return self._captions.vocabulary_size()

    def get_word2id(self):
        return self._captions.word2id

    def get_id2word(self):
        return self._captions.id2word

    def get_caption_txt(self, img_id, cap_offset):
        return self._captions.captions[img_id][cap_offset]

    def get_captions_txt(self, img_id):
        return self._captions.captions[img_id]

    def from_batchlabel2batch_onehot(self, batch_labels):
        batch_onehot = np.zeros(shape=(len(batch_labels), self.vocabulary_size()), dtype=np.float)
        for i, label in enumerate(batch_labels):
            batch_onehot[i, label] = 1.0
        return batch_onehot

