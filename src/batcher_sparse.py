import sys, os
import random
import numpy as np
from mscoco_captions_reader import MSCocoCaptions
from visual_embeddings_reader import VisualEmbeddingsReader

class Batcher:

    def __init__(self,
                 captions_file,
                 visual_file,
                 batch_size,
                 max_vocabulary_size = 10000,
                 word2id=None, id2word=None):
        self._captions = MSCocoCaptions(captions_file, max_vocabulary_size=max_vocabulary_size, word2id=word2id, id2word=id2word)
        self._visual = VisualEmbeddingsReader(visual_file)
        self._batch_size = batch_size
        self.epoch = 0 #automatically increases when all examples have been seen
        self._samples = self._get_samples_coords()
        self._offset = 0

    """
    Return all samples coordinates,
    i.e., [(img_id0,cap0), (img_id0,cap1), ..., (img_id0,cap4), ..., (img_idN,cap4))
    """
    def _get_samples_coords(self):
        all_coords = []
        for img_id in self._visual.visual_embeddings.keys():
            all_coords += [(img_id, cap_id) for cap_id in self.caption_ids(img_id)]
        return all_coords

    def wide1hot(self, img_label, cap_pos):
        caption = self._captions.get_captions(img_label)[cap_pos]
        wide1hot = np.zeros(self.vocabulary_size())
        wide1hot[caption] = 1
        return wide1hot

    def rand_cap(self, img_label):
        return random.choice(range(len(self._captions.captions[img_label])))

    def next(self):
        batch_samples = self._samples[self._offset:self._offset + self._batch_size]
        self._offset += self._batch_size
        if self._offset > len(self._samples):
            self._offset = 0
            self.epoch += 1
            random.shuffle(self._samples)
        if not batch_samples:
            return self.next()

        img_labels, caps_pos, wide_in, wide_out, visual_embeddings = [], [], [], [], []
        for img_id,cap_pos in batch_samples:
            img_labels.append(img_id)
            caps_pos.append(cap_pos)
            wide_in.append(self.wide1hot(img_id, cap_pos))
            wide_out.append(self.wide1hot(img_id, self.rand_cap(img_id)))
            visual_embeddings.append(self._visual.get(img_id))
        return img_labels, caps_pos, wide_in, wide_out, visual_embeddings

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

    def caption_ids(self, img_id):
        return range(self.num_captions(img_id))

    def num_captions(self, img_id):
        return len(self._captions.get_captions(img_id))



