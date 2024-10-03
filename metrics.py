import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score


class Metric(object):
    def __init__(self, output, label):
        self.output = output  # prediction label matric
        self.label = label  # true  label matric

    def accuracy_all(self, thresh=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def accuracy_mean(self, thresh=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        accuracy = np.mean(np.equal(y_true, y_pred))
        return accuracy

    def micfscore(self, thresh=0.5, type='micro'):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        return f1_score(y_pred, y_true, average=type)

    def macfscore(self, thresh=0.5, type='macro'):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        return f1_score(y_pred, y_true, average=type)

    def hamming_distance(self, thresh=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        return hamming_loss(y_true, y_pred)

    def fscore_class(self, type='micro'):
        y_pred = self.output
        y_true = self.label
        return f1_score(np.argmax(y_pred, 1), np.argmax(y_true, 1), average=type)

    def auc(self):
        try:
            auc = roc_auc_score(self.label, self.output, average=None)
            return auc
        except ValueError:
            pass
