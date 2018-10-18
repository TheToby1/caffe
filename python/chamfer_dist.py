import caffe
import numpy as np
from scipy import spatial
TRAIN = 0
TEST = 1

class Chamfer_Dist(caffe.Layer):
    #Setup method
    def setup(self, bottom, top):
        #We want two bottom blobs, the labels and the predictions
        if len(bottom) != 2:
            raise Exception("Wrong number of bottom blobs (reconstruction and original)") 

        #And some top blobs, depending on the phase
        if len(top) != 1:
            raise Exception("Wrong number of top blobs (acc)")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.idx = np.zeros((2, bottom[0].data.shape[0], bottom[0].data.shape[2]), dtype=np.int16)

        # loss output is scalar
        top[0].reshape(1)

    #Forward method
    def forward(self, bottom, top):
        #The order of these depends on the prototxt definition
        recon = bottom[0].data
        orig = bottom[1].data

        batch_size = orig.shape[0]
        num_point = orig.shape[2]

        dist = 0
        for i in range(batch_size):
            tree1 = spatial.KDTree(recon[i].squeeze().T, leafsize=num_point+1)
            tree2 = spatial.KDTree(orig[i].squeeze().T, leafsize=num_point+1)

            distances1, self.idx[0, i] = tree1.query(orig[i].squeeze().T)
            distances2, self.idx[1, i] = tree2.query(recon[i].squeeze().T)

            dist1 = np.sum(np.square(distances1))/2
            dist2 = np.sum(np.square(distances2))/2

            dist += dist1 + dist2
        dist /= batch_size
        #output data to top blob
        top[0].data[...] = [dist]


    def backward(self, top, propagate_down, bottom):

        for i in range(2):
            if not propagate_down[i]:
                continue
            diff = bottom[i].data.copy()
            diff2 = -bottom[i-1].data.copy()

            for j, order in enumerate(self.idx[i-1]):
                diff[j] -= np.swapaxes(bottom[i-1].data[j, :, order, :], 0, 1)

            for j, order in enumerate(self.idx[i]):
                diff2[j] += np.swapaxes(bottom[i].data[j, :, order, :], 0, 1)

            for j, order in enumerate(self.idx[i]):
                for k, pos in enumerate(order):
                    diff[j, :, pos, :] += diff2[j, :, k, :]

            bottom[i].diff[...] = diff / bottom[i].data.shape[0]
