import BEN_filterData as Ben  

fileData  ='/home/ben/AI/AUDIO2/data/fluent_speech_commands_dataset/data/train_data.csv'

keyWord_1 = "Turn on the lights"
keyWord_2 = "Turn off the lights"
keyWord_3 = "Start the music"
keyWord_4 = "Pause the music"
keyWord_5 = "Lights on"
keyWord_6 = "Lights off"

DATA1 =Ben.filterData(fileData , keyWord_1 , usecolsFiter ="transcription" , usecolsPath ="path" )
DATA2 =Ben.filterData(fileData , keyWord_2 , usecolsFiter ="transcription" , usecolsPath ="path" )
DATA3 =Ben.filterData(fileData , keyWord_3 , usecolsFiter ="transcription" , usecolsPath ="path" )
DATA4 =Ben.filterData(fileData , keyWord_4 , usecolsFiter ="transcription" , usecolsPath ="path" )
DATA5 =Ben.filterData(fileData , keyWord_5 , usecolsFiter ="transcription" , usecolsPath ="path" )
DATA6 =Ben.filterData(fileData , keyWord_6 , usecolsFiter ="transcription" , usecolsPath ="path" )

if __name__ == "__main__":

    DATA1.loadFilter()
    #print ( DATA1.loadFilter() )

    DATA2.loadFilter()
    #print ( DATA2.loadFilter() )

    DATA3.loadFilter()
    #print ( DATA3.loadFilter() )

    DATA4.loadFilter()
    #print ( DATA4.loadFilter() )

    DATA1.loadFilter()
    #print ( DATA5.loadFilter() ) 

    DATA5.loadFilter()
    #print ( DATA5.loadFilter() )

    DATA6.loadFilter()
    #print ( DATA6.loadFilter() ) 
