import os
import numpy as np
from io import open

class VisualEmbeddingsReader:
    def __init__(self, visualembeddingsFile, visual_dim=4096):        
        self.visual_dim = visual_dim
        if os.path.exists(visualembeddingsFile+'.npy'):
            print('Loading binary file from %s ...' % (visualembeddingsFile+'.npy'))
            self.visual_embeddings = np.load(visualembeddingsFile+'.npy').item()
            return
        print('Reading images from %s ...' % visualembeddingsFile)
        with open(visualembeddingsFile, "r", buffering=100000000, encoding="utf-8", errors='ignore') as fv:
            self.visual_embeddings = dict()
            for line in fv:
                fields = line.split("\t")
                imgID = int(fields[0])
                embedding = np.zeros(shape=self.visual_dim, dtype=np.float32)
                dim_val = [x.split(':') for x in fields[1].split()]
                for dim, val in dim_val:embedding[int(dim)] = float(val)
                self.visual_embeddings[imgID] = embedding
                if len(self.visual_embeddings) % 1000 == 0: print('\t%d images read' % len(self.visual_embeddings))
        print('Saving binary file %s.npy ...' % visualembeddingsFile)
        np.save(visualembeddingsFile, self.visual_embeddings)

    def get(self, img_label): return self.visual_embeddings[img_label]

    def get_all_vectors(self):
        unzip = zip(*self.visual_embeddings.items())
        img_ids = np.asarray(unzip[0])
        vectors = np.asarray(unzip[1])
        return img_ids, vectors

