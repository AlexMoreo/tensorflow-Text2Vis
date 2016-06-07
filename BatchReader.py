import collections
import random
import bz2
import numpy as np
from io import open


class BatchReader:
    def __init__(self, tr_captionsInputFile, val_captionsInputFile, tr_visualEmbeddingsFile, val_visualEmbeddingsFile,
                 max_vocabulary_size=50000, min_word_occurrences=5, normalize_caption_onehots=False, batch_size=5,
                 random_caption_samples_from_image=1, valid_size=0, test_size=0):
        self.max_vocabulary_size = max_vocabulary_size
        self.min_word_occurrences = min_word_occurrences
        self.normalize_caption_onehots = normalize_caption_onehots
        self.batch_size = batch_size
        self.random_caption_samples_from_image = random_caption_samples_from_image
        captions_tokens, self.captions_orig, words = dict(), dict(), []
        capread = self.captions_reader(tr_captionsInputFile, captions_tokens, self.captions_orig, words)
        capread = self.captions_reader(val_captionsInputFile, captions_tokens, self.captions_orig, words,
                                       previously_read=capread)
        self.captions_indexed, self.vocabulary_size = self.captions_indexer(captions_tokens, words)
        self.visual_reader = VisualEmbeddingsReader(tr_visualEmbeddingsFile)
        self.val_test_reader = VisualEmbeddingsReader(val_visualEmbeddingsFile)
        self.valid_size = valid_size
        self.test_size = test_size
        self.validation_set = None
        self.test_set = None

    def captions_reader(self, captionsFile, captions_tokens, captions_text, words, previously_read=0):
        captions_read = previously_read
        print("Reading captions file <%s>" % captionsFile)
        with bz2.BZ2File(captionsFile, 'r', buffering=10000000) as fin:
            for line in fin:
                line = line.decode("utf-8")
                fields = line.split("\t")
                imageID = int(fields[0])
                tokens = fields[2].split()
                words.extend(tokens)
                if imageID not in captions_tokens:
                    captions_tokens[imageID] = []
                    captions_text[imageID] = []
                captions_tokens[imageID].append(tokens)
                captions_text[imageID].append(fields[2][:-1])
                captions_read += 1
        print("[Done] Read %d images-ids, %d captions %d words, %.2f captions/image, %.2f words/caption" % (
        len(captions_tokens), captions_read, len(words), captions_read * 1.0 / len(captions_tokens),
        len(words) * 1.0 / captions_read))
        return captions_read

    def captions_indexer(self, captions, words):
        print("Building captions indexes")
        count = []
        count.extend(collections.Counter(words).most_common(self.max_vocabulary_size))
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
        for imageID in captions.iterkeys():
            captions_indexed[imageID] = []
            for caption in captions[imageID]:
                caption_indexed = []
                for word in caption:
                    if word in dictionary:
                        caption_indexed.append(dictionary[word])
                captions_indexed[imageID].append(caption_indexed)
        print("Vocabulary (min_word_occurrences>=%d) has length %d" % (self.min_word_occurrences, vocabulary_size))
        return captions_indexed, vocabulary_size

    def index2onehot(self, indexed_captions):
        one_hots = np.zeros(shape=(len(indexed_captions), self.vocabulary_size))
        for row, indexed_caption in enumerate(indexed_captions):
            for col in indexed_caption:
                one_hots[row, col] = 1.0 / (len(
                    indexed_caption) if self.normalize_caption_onehots else 1.0)  # to make them sum up to 1, so as to be interpreted as a prob distribution
        return one_hots

    def nextBatch(self, batch_size=-1, img_reader=None, radom_cap=True, random_caption_samples_from_image=None):
        if batch_size == -1:
            batch_size = self.batch_size
        if img_reader == None:
            img_reader = self.visual_reader
        if random_caption_samples_from_image == None:
            random_caption_samples_from_image = self.random_caption_samples_from_image
        inputCaption, outputCaption, outputVisual = [], [], []
        inputCaptionOffset, outputCaptionOffset, imageLabel = [], [], []

        for _ in range(batch_size / random_caption_samples_from_image):
            imgID, imgVE = img_reader.getnext()
            num_img_captions = len(self.captions_indexed[imgID])
            for cappos in range(random_caption_samples_from_image):
                capInPos = int(
                    random.randint(0, num_img_captions - 1)) if radom_cap == True else cappos % num_img_captions
                # capInPos  = 0 if radom_cap==True else cappos%num_img_captions
                capIn = self.captions_indexed[imgID][capInPos]
                capOutPos = int(
                    random.randint(0, num_img_captions - 1)) if radom_cap == True else cappos % num_img_captions
                capOut = self.captions_indexed[imgID][capOutPos]  # can be the same as the input one
                inputCaption.append(capIn)
                outputCaption.append(capOut)
                outputVisual.append(imgVE)
                inputCaptionOffset.append(capInPos)
                outputCaptionOffset.append(capOutPos)
                imageLabel.append(imgID)

        return self.index2onehot(inputCaption), self.index2onehot(
            outputCaption), outputVisual, inputCaptionOffset, outputCaptionOffset, imageLabel

    def getValidationSet(self):
        # reads the validation set only when requested; reads only 1 input_captions-image combination
        if self.validation_set == None:
            print("Reading the validation set [%d images]" % self.valid_size)
            self.validation_set = self.nextBatch(self.valid_size, self.val_test_reader, radom_cap=False,
                                                 random_caption_samples_from_image=1) if self.valid_size > 0 else []
        return self.validation_set

    def getTestSet(self):
        # reads the test set only when requested; reads all the input_captions-image combinations (5 in MSCoco)
        if self.test_set == None:
            print("Reading the test set [%d images]" % self.test_size)
            _ = self.getValidationSet()  # first assures that the validation set has been read before
            self.test_set = self.nextBatch(self.test_size, self.val_test_reader, radom_cap=False,
                                           random_caption_samples_from_image=1) if self.test_size > 0  else []
        return self.test_set


class VisualEmbeddingsReader:
    def __init__(self, visualembeddingsFile):
        self.images_ids = []
        self.visual_embeddings = dict()
        self.all_read = False
        self.visualembeddingsFile = open(visualembeddingsFile, "r", buffering=100000000, encoding="utf-8",
                                         errors='ignore')

    def getnext(self):
        if self.all_read == False:
            try:
                line = self.visualembeddingsFile.next()
                fields = line.split("\t")
                imgID = int(fields[0].split("_")[2])
                veData = list(map(lambda x: max(0.0, float(x)), fields[1].split()))
                ar = np.array(veData, dtype=np.float32)
                self.images_ids.append(imgID)
                self.visual_embeddings[imgID] = ar
                return imgID, ar
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
