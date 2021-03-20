import librosa
import numpy as np
import librosa
#import librosa.display
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
#import array as arr
#import csv
# from pydub import AudioSegment
# sound = AudioSegment.from_mp3("D:/work/AI.Edge-Audio2/AUDIO_V2/dataToTest/Go.mp3")
# sound.export("D:/work/AI.Edge-Audio2/AUDIO_V2/dataToTest/Go.wav", format="wav")

SAMPLES_TO_CONSIDER = 22050
# file = "D:/work/AI.Edge-Audio2/AUDIO_V2/dataSet/down/0a9f9af7_nohash_1.wav"
#
file = "/home/ben/AI/AUDIO2/source/makerMFFCs/output (mp3cut.net).wav"
# file = "D:/work/AI.Edge-Audio2/AUDIO_V2/off_2.wav"
#file = "D:/work/Python/ProcessFileWav/output.wav"
# file = "D:/work/AI.Edge-Audio2/AUDIO_V2/dataSet/off/0ab3b47d_nohash_1.wav"
# file = "D:/work/AI.Edge-Audio2/AUDIO_V2/dataSet/yes/0a9f9af7_nohash_2.wav"
#Datapath = 'D:\work\AI.Edge-Audio2\Python\dataToTest.json'
fileDataJson = "dataToTest.json"
def preprocessDataset ( file , fileDataJson , num_mfcc=13, n_fft=2048, hop_length=512):
    data = {
    	"data":[]
    }
    signal, sample_rate = librosa.load(file)
    print(signal)
    
    if len(signal) >= SAMPLES_TO_CONSIDER:
    	# ensure consistency of the length of the signal
    	signal = signal[:SAMPLES_TO_CONSIDER]
    
    	# extract MFCCs
    	MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
    								 hop_length=hop_length)
    	print(MFCCs.T.shape)
    	data["data"].append(MFCCs.T.tolist())
    with open(fileDataJson , "w") as js:
    	json.dump(data, js , indent = 1 )
    
    return MFCCs.T
preprocessDataset ( file , fileDataJson , num_mfcc=13, n_fft=2048, hop_length=512)




