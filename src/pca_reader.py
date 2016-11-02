import numpy as np
from io import open

class PCAprojector():
    def __init__(self, mean_file, eigen_file, num_dims, num_eig=256):
        self.num_dims=num_dims
        self.read_mean(mean_file)
        self.read_eigen(eigen_file, num_eig)

    def read_mean(self, mean_file):
        with open(mean_file, "r") as mean_row:
            mean_str = mean_row.readlines()
            assert(len(mean_str)==1)
            self.mean = self.read_row(mean_str[0])
            print("mean read with shape %s" % str(self.mean.shape))

    def read_eigen(self, eigen_file, num_eig):
        self.num_eigen=num_eig
        self.eigen = np.empty([self.num_eigen, self.num_dims])
        current_row = 0
        with open(eigen_file, "r") as eigen_rows:
            vectors = eigen_rows.readlines()
            assert (len(vectors) >= self.num_eigen)
            for vec in vectors[:self.num_eigen]:
                self.eigen[current_row]=self.read_row(vec)
                current_row += 1
        print("eigen read with shape %s" % str(self.eigen.shape))

    def read_row(self,row):
        vals = row.split()
        assert (len(vals) == self.num_dims)
        return np.array([float(x) for x in vals])

    def project(self, matrix):
        return np.dot(matrix - self.mean, self.eigen.transpose())

