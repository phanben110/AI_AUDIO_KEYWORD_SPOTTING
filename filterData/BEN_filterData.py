import pandas as pd 
import numpy as np  
fileData  ='/home/ben/AI/AUDIO2/data/fluent_speech_commands_dataset/data/train_data.csv'

class filterData: 
    #number= [0] 
    printDataPath = False 
    #data = None 
    def __init__(self, fileData, keyWord , usecolsFiter = "transcription", usecolsPath = "path") :
        self.fileData = fileData 
        self.keyWord = keyWord  
        self.usecolsFiter = usecolsFiter 
        self.usecolsPath = usecolsPath
        self.number = [0] 
        self.printDataPath = False    
        self.data = [None] 
        """
        paramiter fileData is String file data 
        paramiter keyword is String 
        paramiter usecolsFiter is a transcription you want to fiter  
        paramiter usecolsPath is a path want to acces 

        """
    def readAndFilter(self): 
        data = pd.read_csv(self.fileData , usecols=[self.usecolsFiter])
        data = np.array(data) 
        shape  = data.shape[0]
        #print(f"dataset is {shape} value") 
        count =0    
        for i in range(shape):
            if data[i] == self.keyWord:
                self.number.append(i)
                count +=1 
        print ( f"Key word [ {self.keyWord} ]  have {count} data") 
        return self.number


    def printDataPath(self):
        self.printDataPath = True 
        return self.printDataPath

    def readPath(self ):
        dataPath = pd.read_csv(self.fileData, usecols =[self.usecolsPath] ) 
        dataPath = np.array(dataPath) 
        dataPath = np.array(dataPath) 
        if self.printDataPath == True: 
            print ("================begin===============")
        else :
            pass 
        for i in range (len(self.number)) :
            if i != 0 :
                self.data.append(dataPath[self.number[i]])   
            else :
                pass
        if self.printDataPath == True:
            print(self.data)
            print ("================end===============")

    def loadFilter(self): 
        self.readAndFilter()  
        self.readPath()
        return self.data 
        
