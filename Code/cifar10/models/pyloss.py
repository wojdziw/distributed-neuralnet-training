import caffe
import numpy as np


class EuclideanLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)


    def forward(self, bottom, top):
        self.diff[...] = (bottom[0].data - bottom[1].data) / np.sqrt(np.sum(bottom[0].data**2)+np.sum(bottom[1].data**2))
        # because I'm squaring the denominator of the line above, I need to multiply it again here
        #top[0].data[...] = np.sqrt(np.sum(self.diff**2)) * np.sqrt(np.sum(bottom[0].data**2)+np.sum(bottom[1].data**2)) / bottom[0].num / 2.
        top[0].data[...] = np.sum(self.diff**2) * np.sqrt(np.sum(bottom[0].data**2)+np.sum(bottom[1].data**2)) / bottom[0].num / 2.
    '''
    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.
    '''


    def backward(self, top, propagate_down, bottom):
        # range 2 because I have two bottoms
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
