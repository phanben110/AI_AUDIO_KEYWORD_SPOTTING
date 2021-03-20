import librosa
import os
import json
import BEN_filterData as Ben

fileData  ='/home/ben/ai/audio/audioEdge/data/fluent_speech_commands_dataset/data/train_data.csv'

pathDataBefore  ='/home/ben/ai/audio/audioEdge/data/fluent_speech_commands_dataset/'

keyWord_1 = "Turn on the lights"
keyWord_2 = "Turn off the lights"
keyWord_3 = "Start the music"
keyWord_4 = "Pause the music"
keyWord_5 = "Lights on"
keyWord_6 = "Lights off"

DATA1 =Ben.filterData(fileData , keyWord_1 ,pathDataBefore , usecolsFiter ="transcription" , usecolsPath ="path" )
DATA2 =Ben.filterData(fileData , keyWord_2 ,pathDataBefore , usecolsFiter ="transcription" , usecolsPath ="path" )
DATA3 =Ben.filterData(fileData , keyWord_3 ,pathDataBefore , usecolsFiter ="transcription" , usecolsPath ="path" )
DATA4 =Ben.filterData(fileData , keyWord_4 ,pathDataBefore , usecolsFiter ="transcription" , usecolsPath ="path" )
DATA5 =Ben.filterData(fileData , keyWord_5 ,pathDataBefore , usecolsFiter ="transcription" , usecolsPath ="path" )
DATA6 =Ben.filterData(fileData , keyWord_6 ,pathDataBefore , usecolsFiter ="transcription" , usecolsPath ="path" )
DATA = [] 
DATA.append ( DATA1.loadFilter())
DATA.append ( DATA2.loadFilter())
DATA.append ( DATA3.loadFilter())
DATA.append ( DATA4.loadFilter())
DATA.append ( DATA5.loadFilter())
DATA.append ( DATA6.loadFilter())

DATASET_PATH = "dataSet"
JSON_PATH = "dataKeyWordV1.json"
SAMPLES_TO_CONSIDER = 22050  # 1 sec worth of sound đây là lấy mẫu trong vòng 1 giây

LABEL  = [keyWord_1 , keyWord_2 , keyWord_3 , keyWord_4 , keyWord_5 , keyWord_6]


def preprocessDataset( jsonPath, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from music dataset and saves them into a json file.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """

    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "nameLabels": [],    
        "labels": [],
        "MFCCs": [],
        "files": []
    } 

    for i in range(len(DATA)) :
        print(i) 
        for j in range(len(DATA[i])):
                
            signal, sampleRate = librosa.load(DATA[i][j])
            # load audio file and slice it to ensure length consistency among different files
            
            # drop audio files with less than pre-decided number of samples
            if len(signal) >= SAMPLES_TO_CONSIDER:
            
                # ensure consistency of the length of the signal
                signal = signal[:SAMPLES_TO_CONSIDER]
            
                # extract MFCCs
                MFCCs = librosa.feature.mfcc(signal, sampleRate, n_mfcc=num_mfcc, n_fft=n_fft,
                                             hop_length=hop_length)
            
                # store data for analysed track
                data["MFCCs"].append(MFCCs.T.tolist())
                data["nameLabels"].append(LABEL[i])
                data["labels"].append(i) 
                data["files"].append(DATA[i][j])
                print("{}: {} so data : {} ".format(DATA[i][j], i, j))
        print (f"done {i}")                 
    # save d#ata in json file
    with open(jsonPath, "w") as fp:
        json.dump(data, fp, indent=3)


if __name__ == "__main__":
    preprocessDataset(JSON_PATH)
    print("======================= done =======================")


