import os
import sys
from scipy.io import wavfile
import scipy
from pydub import AudioSegment
import matplotlib.pyplot as plt
import pylab
from matplotlib.pyplot import specgram
import cPickle as pickle
import json


INIT_DIR = "/home/krishna/Documents/workspace/MusicGenreDetection/genres/"
DATA_DIR = "/home/krishna/Documents/workspace/MusicGenreDetection/data/"
TEMP_TEST_FILE = "/home/krishna/Documents/workspace/MusicGenreDetection/data/blues/blues.00000.wav"
TEMP_TEST_FILE1 = "/home/krishna/Documents/workspace/MusicGenreDetection/data/rock/rock.00000.wav"
TEMP_TEST_FILE_BLUES = "/home/krishna/Documents/workspace/MusicGenreDetection/data/blues/blues.00020.wav"
TEMP_TEST_FILE_METAL = "/home/krishna/Documents/workspace/MusicGenreDetection/data/metal/metal.00001.wav"
GENRE_CLASSES = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
TEST_DIR = "/home/krishna/Documents/workspace/MusicGenreDetection/test/"
ROOT_DIR = "/home/krishna/Documents/workspace/MusicGenreDetection/"

path = "/home/krishna/Documents/workspace/MusicGenreClassification/data_dump/"
data_dump = {}  
for eachdir in os.listdir(DATA_DIR):
    genre = eachdir
    data_dump[genre]=[]
    print genre
    current_dir = DATA_DIR+eachdir+"/"
    for eachdir in os.listdir(current_dir):
        print "processing "+eachdir
        sample_rate, X = wavfile.read(current_dir+eachdir)
        fft_features = abs(scipy.fft(X))
        data_dump[genre].append(fft_features)
os.chdir(path)
pickle.dump(data_dump,open('data.p','wb'))