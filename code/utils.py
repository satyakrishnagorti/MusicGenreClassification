import os
import sys
from scipy.io import wavfile
import scipy
import numpy
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib import pylab
import pylab
from matplotlib.pyplot import specgram
import cPickle as pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import glob
from constants import FFT_PATH,INIT_DIR,DATA_DIR,GENRE_CLASSES,ROOT_DIR,TEST_DIR,FFT_TEST_PATH


def write_fft(path,fft_features,test = False):
  """
    saving each song's fft in a file
  """
  if test == True:
    fft_path = FFT_TEST_PATH
  else:
    fft_path = FFT_PATH
  base,ext = os.path.splitext(path)
  song = base.split('/')[-1]
  print "writing:",song
  #numpy.save(FFT_PATH+song+".fft",fft_features)
  numpy.save(fft_path+song+".fft",fft_features)
  print "Saved:"+song

def generate_fft(path,test = False):
  """ 
    generating each song's fft
  """
  sample_rate, X = wavfile.read(path)
  fft_features = abs(scipy.fft(X)[:60000])
  if test==True:
    print "writing test data"
    write_fft(path,fft_features,True)
  else:
    write_fft(path,fft_features)

def read_fft(path):
  """
    reading each all fft files and returning it along with it's corresponding genre
  """
  x = []
  y = []
  for f in glob.glob(path+"*.fft*"):
    print "reading:",f
    fft = numpy.load(f)
    print fft
    #x.append(np.mean(fft[int(len(fft)/10): int(len(fft*9)/10)]))
    print "length:",len(fft)
    #print fft[int(len(fft)/10):int(len(fft*9)/10)]
    #x.append(fft[int(len(fft)/10):int(len(fft*9)/10)])
    x.append(fft[:650000])
    #print x
    genre = f.split(".")[0]
    genre = genre.split("/")[-1]
    print "genre:",genre
    y.append(genre)
  return np.array(x),np.array(y)

def plot_roc(auc_score, name, tpr, fpr, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.plot(fpr, tpr)
    pylab.fill_between(fpr, tpr, alpha=0.5)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('ROC curve (AUC = %0.2f) / %s' %
                (auc_score, label), verticalalignment="bottom")
    pylab.legend(loc="lower right")
    filename = name.replace(" ", "_")
    pylab.savefig(
        os.path.join(CHART_DIR, "roc_" + filename + ".png"), bbox_inches="tight")

if __name__ == '__main__':
  """ run this first to generate fft data
  for eachdir in os.listdir(DATA_DIR):
    genre = eachdir
    print genre
    current_dir = DATA_DIR+eachdir+"/"
    for eachdir in os.listdir(current_dir):
        print "processing "+eachdir
        generate_fft(current_dir+eachdir)
  """
  for eachdir in os.listdir(TEST_DIR):
    genre = eachdir
    print genre
    current_dir = TEST_DIR + eachdir+"/"
    for eachdir in os.listdir(current_dir):
      print "processing " + eachdir
      generate_fft(current_dir+eachdir,test= True)
  #read_fft(FFT_PATH)