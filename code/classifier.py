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
from constants import FFT_PATH,INIT_DIR,DATA_DIR,GENRE_CLASSES,ROOT_DIR,TEST_DIR,DUMP_PATH
from utils import read_fft


def classification_model():
  clf= LogisticRegression()
  return clf

def read_dump(data_dump_path):
  print 'loading'
  data_dump = pickle.load(open(data_dump_path,'rb'))
  print 'Done'
  x=[]
  y=[]
  for key in data_dump: 
    for each in data_dump[key]:
      x.append(each[:650000                                                                                                                                                                                                                                                                           ])
      print each[0],each[1],each[2]
      y.append(key)
  return x,y 

def train_model(x,y):
  clf = LogisticRegression()
  clfs = []
  clf.fit(x,y)
  print "Writing model"
  joblib.dump(clf,DUMP_PATH+"clf.pkl")
  print "Done"


if __name__ == '__main__':
  import timeit
  start = timeit.default_timer()
  print "reading x,y np array"
  x,y = read_fft(FFT_PATH)
  print x,y
  print x.shape, y.shape
  print "done"
  print 'running classifier'
  clf = classification_model()
  train_model(x,y)
  stop = timeit.default_timer()
  print "Total time (s) = ", (stop - start)
