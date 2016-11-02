import sys, os
import random
import numpy as np
import bz2
sys.path.append(os.getcwd())
from nltk.stem import WordNetLemmatizer
from visual_embeddings_reader import VisualEmbeddingsReader

class Batcher:

    def __init__(self,
                 captions_file,
                 visual_file,
                 we_dim,
                 batch_size,
                 lemmatize,
                 model):
        self._captions = self.read_mscococaptions(captions_file, lemmatize)
        self._visual = VisualEmbeddingsReader(visual_file)
        self._batch_size = batch_size
        self.we_dim = we_dim
        self.epoch = 0 #automatically increases when all examples have been seen
        self._model=model
        self._samples = self._get_samples_coords()
        self._offset = 0

    """
    Return all samples coordinates,
    i.e., [(img_id0,cap0), (img_id0,cap1), ..., (img_id0,cap4), ..., (img_idN,cap4))
    """
    def _get_samples_coords(self):
        all_coords=[]
        for img_id in self._visual.visual_embeddings.keys():
            all_coords+=[(img_id,cap_id) for cap_id in self.caption_ids(img_id)]
        return all_coords

    def next(self):
        batch_samples = self._samples[self._offset:self._offset + self._batch_size]
        self._offset += self._batch_size
        if self._offset > len(self._samples):
            self._offset = 0
            self.epoch += 1
            random.shuffle(self._samples)
        if not batch_samples:
            return self.next()

        img_labels,caps_pos, pooled_embeddings, visual_embeddings = [], [], [], []
        for img_id,cap_pos in batch_samples:
            img_labels.append(img_id)
            caps_pos.append(cap_pos)
            pooled_embeddings.append(self.pool_sentence(self._captions[img_id][cap_pos]))
            visual_embeddings.append(self._visual.get(img_id))

        return img_labels, caps_pos, pooled_embeddings, visual_embeddings

    def get_caption_txt(self, img_id, cap_offset):
        return self._captions[img_id][cap_offset]

    def get_captions_txt(self, img_id):
        return self._captions[img_id]

    def num_captions(self, img_id):
        return len(self._captions[img_id])

    def caption_ids(self, img_id):
        return range(self.num_captions(img_id))

    def pool_sentence(self, sentence):
        pooled = np.zeros(self.we_dim)
        items = 0
        for w in sentence.split():
            if w in self._model:
                pooled += self._model[w]
                items += 1
        if not items:
            print('warning: no model found for sentence %s.' %sentence)
        return pooled / items if items > 0 else 1

    def read_mscococaptions(self, captions_file, lemmatize=False):
        lemmatizer = WordNetLemmatizer() if lemmatize else None
        print("Reading captions file <%s>" % captions_file)
        captions = dict()
        with bz2.BZ2File(captions_file, 'r', buffering=10000000) as fin:
            for line in fin:
                line = line.decode("utf-8")
                fields = line.split("\t")
                imageID = int(fields[0])
                sentence = fields[2][:-1].lower()
                if lemmatize:
                    sentence = lemmatizer.lemmatize(sentence)
                if imageID not in captions:
                    captions[imageID] = []
                captions[imageID].append(sentence)
        return captions

