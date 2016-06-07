import collections
import random
import bz2
import sys
import os
import math
import numpy as np
from io import open

class BatchReader:
    def __init__(self, tr_captions_file, val_captions_file, tr_visual_file, val_visual_file=None, test_visual_file=None,
                 max_vocabulary_size=50000, min_word_occurrences=5, normalize_caption_onehots=False, batch_size=5,
                 random_caption_samples_from_image=1, valid_size=0, test_size=0, random_index_dims=-1, visual_dim=4096,
                 lightweight_ri=False):
        self.batch_size = batch_size
        self.random_caption_samples_from_image = random_caption_samples_from_image
        self.caption_reader = CaptionReader(tr_captions_file, val_captions_file,
                                            max_vocabulary_size=max_vocabulary_size,
                                            min_word_occurrences=min_word_occurrences,
                                            normalize_caption_onehots=normalize_caption_onehots)
        self.tr_visual_reader   = VisualEmbeddingsReader(tr_visual_file, visual_dim)
        self.val_visual_reader  = VisualEmbeddingsReader(val_visual_file, visual_dim)
        self.test_visual_reader = VisualEmbeddingsReader(test_visual_file, visual_dim)
        self.valid_size = valid_size
        self.test_size = test_size
        self.validation_set = None
        self.test_set = None

        if random_index_dims == -1:
            captions_dimensions = self.caption_reader.vocabulary_size
            k=1
        else:
            captions_dimensions = random_index_dims
            k=(2 if lightweight_ri else int(captions_dimensions*0.01))
        self.caption_encoder = CaptionEncoder(dims=captions_dimensions, k=k)

    def text_embedding_size(self):
        return self.caption_encoder.dims

    def next_batch(self, batch_size=-1, img_reader=None, random_cap=True, random_caption_samples_from_image=None):
        if batch_size == -1:
            batch_size = self.batch_size
        if img_reader is None:
            img_reader = self.tr_visual_reader
        if random_caption_samples_from_image is None:
            random_caption_samples_from_image = self.random_caption_samples_from_image
        in_cap_raw, out_cap_raw, out_vis_code = [], [], []
        in_cap_offset, out_cap_offset, img_label = [], [], []

        for _ in range(batch_size / random_caption_samples_from_image):
            imgID, imgVE = img_reader.getnext()
            num_img_captions = len(self.caption_reader.captions_indexed[imgID])
            for cappos in range(random_caption_samples_from_image):
                capInPos = int(random.randint(0, num_img_captions - 1)) if random_cap == True else cappos % num_img_captions
                # capInPos  = 0 if random_cap==True else cappos%num_img_captions
                capIn = self.caption_reader.captions_indexed[imgID][capInPos]
                capOutPos = int(random.randint(0, num_img_captions - 1)) if random_cap == True else cappos % num_img_captions
                capOut = self.caption_reader.captions_indexed[imgID][capOutPos]  # can be the same as the input one
                in_cap_raw.append(capIn)
                out_cap_raw.append(capOut)
                out_vis_code.append(imgVE)
                in_cap_offset.append(capInPos)
                out_cap_offset.append(capOutPos)
                img_label.append(imgID)

        in_cap_code, out_cap_code = self.caption_encodding(in_cap_raw), self.caption_encodding(out_cap_raw)
        return in_cap_code, out_cap_code, out_vis_code, in_cap_offset, out_cap_offset, img_label

    def getValidationSet(self):
        # reads the validation set only when requested; reads only 1 input_captions-image combination
        if self.validation_set is None:
            print("Reading the validation set [%d images]" % self.valid_size)
            self.validation_set = self.next_batch(batch_size=self.valid_size, img_reader=self.val_visual_reader, random_cap=False,
                                                 random_caption_samples_from_image=1) if self.valid_size > 0 else []
        return self.validation_set

    def getTestSet(self):
        # reads the test set only when requested; reads all the input_captions-image combinations (5 in MSCoco)
        if self.test_set is None:
            print("Reading the test set [%d images]" % self.test_size)
            self.test_set = self.next_batch(batch_size=self.test_size, img_reader=self.test_visual_reader, random_cap=False,
                                           random_caption_samples_from_image=1) if self.test_size > 0  else []
        return self.test_set

    def caption_encodding(self, indexed_captions):
        r_projection = np.zeros(shape=(len(indexed_captions), self.text_embedding_size()))
        for row, indexed_caption in enumerate(indexed_captions):
            for col in indexed_caption:
                for (ri_dim, ri_val )in self.caption_encoder.get_indexing(col):
                    r_projection[row, ri_dim] += ri_val
        return r_projection


class CaptionReader:
    def __init__(self, tr_captions_file, val_captions_file, max_vocabulary_size=50000, min_word_occurrences=5,
                 normalize_caption_onehots=False):
        self.max_vocabulary_size = max_vocabulary_size
        self.min_word_occurrences = min_word_occurrences
        self.normalize_caption_onehots = normalize_caption_onehots
        self.captions_tokens, self.captions_orig, self.words = dict(), dict(), []
        capread = self.read_captions(tr_captions_file)
        capread = self.read_captions(val_captions_file, previously_read=capread)
        self.captions_indexed, self.vocabulary_size = self.captions_indexer()

    def read_captions(self, captions_file, previously_read=0):
        captions_read = previously_read
        print("Reading captions file <%s>" % captions_file)
        with bz2.BZ2File(captions_file, 'r', buffering=10000000) as fin:
            for line in fin:
                line = line.decode("utf-8")
                fields = line.split("\t")
                imageID = int(fields[0])
                sentence = fields[2][:-1]
                tokens = sentence.split()
                self.words.extend(tokens)
                if imageID not in self.captions_tokens:
                    self.captions_tokens[imageID], self.captions_orig[imageID] = [], []
                self.captions_tokens[imageID].append(tokens)
                self.captions_orig[imageID].append(sentence)
                captions_read += 1
                if captions_read % 50000 == 0:
                    print("\tread %d..." % captions_read)
        print("[Done] Read %d images-ids, %d captions %d words, %.2f captions/image, %.2f words/caption" % (
            len(self.captions_tokens), captions_read, len(self.words), captions_read * 1.0 / len(self.captions_tokens),
            len(self.words) * 1.0 / captions_read))
        return captions_read

    def captions_indexer(self):
        print("Building captions indexes")
        count = []
        count.extend(collections.Counter(self.words).most_common(self.max_vocabulary_size))
        if self.min_word_occurrences != -1:
            for i, (_, occ) in enumerate(count):
                if occ < self.min_word_occurrences:
                    count = count[:i]
                    break
        vocabulary_size = len(count)
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        captions_indexed = dict()
        for imageID in self.captions_tokens.iterkeys():
            captions_indexed[imageID] = []
            for caption in self.captions_tokens[imageID]:
                caption_indexed = []
                for word in caption:
                    if word in dictionary:
                        caption_indexed.append(dictionary[word])
                captions_indexed[imageID].append(caption_indexed)
        print("Vocabulary (min_word_occurrences>=%d) has length %d" % (self.min_word_occurrences, vocabulary_size))
        return captions_indexed, vocabulary_size


class CaptionEncoder:
    def __init__(self, dims, k=1):
        self.dims=dims
        self.dim_index = dict()
        self.k=k
        self.val=1.0/math.sqrt(self.k*1.0)
        self.signed = [self.val,-self.val]

    def get_indexing(self, v):
        if v not in self.dim_index:
            self.dim_index[v]=[(v%self.dims, self.val)]+[(random.randint(0,self.dims-1), random.choice(self.signed) ) for _ in range(self.k-1)]
            #print("Vector for %d is %s" % (v, str(self.dim_index[v])))
        return self.dim_index[v]


class VisualEmbeddingsReader:
    def __init__(self, visualembeddingsFile, visual_dim=4096):
        self.images_ids = []
        self.visual_embeddings = dict()
        self.all_read = False
        self.visual_dim=visual_dim
        self.visualembeddingsFile = open(visualembeddingsFile, "r", buffering=100000000, encoding="utf-8",
                                         errors='ignore') if not visualembeddingsFile is None else None

    def getnext(self):
        if not self.all_read:
            try:
                line = self.visualembeddingsFile.next()
                fields = line.split("\t")
                imgID = int(fields[0])
                #veData = list(map(lambda x:float(x), fields[1].split()))
                embedding = np.zeros(shape=(self.visual_dim), dtype=np.float32)
                dim_val = [x.split(':') for x in fields[1].split()]
                for dim,val in dim_val:
                    embedding[int(dim)]=float(val)
                self.images_ids.append(imgID)
                self.visual_embeddings[imgID] = embedding
                return imgID, embedding
            except StopIteration as e:
                self.visualembeddingsFile.close()
                print("All training images have been seen. Reinit training.")
                self.image_offset = 0
                self.all_read = True
                return self.getnext()
        else:
            if self.image_offset == 0:
                print("Shuffling images...")
                random.shuffle(self.images_ids)
            imgID = self.images_ids[self.image_offset]
            ar = self.visual_embeddings[imgID]
            self.image_offset = (self.image_offset + 1) % len(self.images_ids)
            return imgID, ar
