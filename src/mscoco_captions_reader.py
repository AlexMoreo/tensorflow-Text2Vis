import sys
import collections
import bz2
import numpy as np
from nltk.stem import WordNetLemmatizer

class MSCocoCaptions(object):

    def __init__(self, captions_file, max_vocabulary_size=10000, word2id=None, id2word=None, lemma=False):
        self.sos_char = '[SOS]' #start of sentence
        self.eos_char = '[EOS]' #end of sentence
        self.unk_word = '[UNK]' #unknow word
        self.pad_word = '[PAD]' #padding token
        self.captions, text = self._read_captions(captions_file, lemma)
        self.word2id=word2id
        self.id2word=id2word        
        if self.word2id is None or self.id2word is None:
            self.word2id, self.id2word = self._build_dataset(max_vocabulary_size, text)
        self._index_captions()

    def _from_caption2ids(self, caption):
        return [self._fromword2id(x) for x in ((self.sos_char + ' ') + caption + (' ' + self.eos_char) * 2).split()]

    def _from_ids2caption(self, ids):
        return ' '.join([self._fromid2word(x) for x in ids])

    def _fromword2id(self, word):
        return self.word2id[word] if word in self.word2id else self.word2id[self.unk_word]

    def _fromid2word(self, wordid):
        return self.id2word[wordid] if wordid in self.id2word else self.unk_word

    def _index_captions(self):
        self.indexed_captions=dict()
        for imageID in self.captions.keys():
            self.indexed_captions[imageID]=[]
            for cap_i in self.captions[imageID]:
                cap_ids = self._from_caption2ids(cap_i)
                self.indexed_captions[imageID].append(cap_ids)

    def _build_dataset(self, max_vocabulary_size, text):
        sys.stdout.write('Building indexes...')
        sys.stdout.flush()
        tokens = text.split()
        count = [[self.unk_word, -1], [self.sos_char, -1], [self.eos_char, -1], [self.pad_word, -1]]
        count.extend(collections.Counter(tokens).most_common(max_vocabulary_size-len(count)))
        word2id = dict()
        for word, _ in count:
            word2id[word] = len(word2id)
        id2word = dict(zip(word2id.values(), word2id.keys()))
        print("[Done]")
        return word2id, id2word

    def vocabulary_size(self):
        return len(self.id2word)

    def _read_captions(self, captions_file, lemmatize=False):
        lemmatizer = WordNetLemmatizer() if lemmatize else None
        print("Reading captions file <%s>" % captions_file)
        captions=dict()
        text = []
        with bz2.BZ2File(captions_file, 'r', buffering=10000000) as fin:
            for line in fin:
                line = line.decode("utf-8")
                fields = line.split("\t")
                imageID = int(fields[0])
                sentence = fields[2][:-1].lower()
                if lemmatize:
                    sentence = lemmatizer.lemmatize(sentence)
                if imageID not in captions:
                    captions[imageID]=[]
                captions[imageID].append(sentence)
                text.append(sentence)
        text=' '.join(text)
        return captions, text

    def _words(self, probabilities):
        """Turn a 1-hot encoding or a probability distribution over the possible
        characters back into its (most likely) word representation."""
        return [self.fromid2word(c) for c in np.argmax(probabilities, 1)]


    def num_images(self):
        return len(self.indexed_captions)

    def get_image_ids(self):
        return self.indexed_captions.keys()

    def get_captions(self, img_id):
        return self.indexed_captions[img_id]

    def get_captions_txt(self, img_id):
        return self.captions[img_id]

    def get_caption_txt(self, img_id, cap_pos):
        return self.captions[img_id][cap_pos]


    def get_pad(self):
        return self._fromword2id(self.pad_word)