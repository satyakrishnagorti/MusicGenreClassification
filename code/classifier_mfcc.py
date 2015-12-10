import os
import sys
from scipy.io import wavfile
import scipy
import numpy
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import pylab
from matplotlib.pyplot import specgram
import cPickle as pickle
import json
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.externals import joblib
import glob
from constants import INIT_DIR,DATA_DIR,GENRE_CLASSES,ROOT_DIR,TEST_DIR,DUMP_PATH,MFCC_PATH,MFCC_TEST_PATH
from utils import read_fft
from mfcc_utils import read_mfcc  
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def classification_model(name):
  print "name is:"+name
  if name=="logreg":
    print "Logistical Regression"
    clf= LogisticRegression()
  elif name=="knn":
    print "kth Nearest Neighbor"
    clf = KNeighborsClassifier(n_neighbors=5)
  elif name=="svm":
    print "SVM"
    clf = svm.SVC()
  elif name=="onevsrest":
    clf = OneVsRestClassifier(svm.SVC())
  elif name=="decisiontree":
    print "Decision Tree"
    clf = DecisionTreeClassifier(max_depth=5)
  else:
    print "default : logistical regression"
    clf = LogisticRegression()
    return clf
  return clf

def train_model(x,y):
  clf = classification_model("decisiontree")
  clfs = []
  print x.shape, y.shape
  clf.fit(x,y)
  print "Writing model"
  joblib.dump(clf,DUMP_PATH+"clf.pkl")
  print "Done"
  return clf

def myplot(x,y,x_test,y_test,y_pred):
  oc_scores = defaultdict(list)
  tprs = defaultdict(list)
  fprs = defaultdict(list)
  labels = np.unique(y)
  for label in labels:
    y_label


if __name__ == '__main__':
  import timeit
  start = timeit.default_timer()
  print "reading x,y np array"
  x,y = read_mfcc(MFCC_PATH)

  print "printing y:"
  print y
  x_test,y_test = read_mfcc(MFCC_TEST_PATH)
  print "printing y_test"
  print y_test, y_test.shape
  print 'running classifier'
  print x.shape
  clf = train_model(x,y)
  y_pred = clf.predict(x_test)
  print "printing y_pred"
  print y_pred
  print len(y_pred)
  score = accuracy_score(y_test,y_pred)
  print score
  stop = timeit.default_timer()
  print "Total time (s) = ", (stop - start)