import winsound
# from playsound import playsound
freq = 100
dur = 50
  
# loop iterates 5 times i.e, 5 beeps will be produced.
for i in range(0, 5):    
    winsound.Beep(freq, dur)    
    freq+= 100
    dur+= 50

# winsound.PlaySound('alarm.mp3', winsound.SND_FILENAME)  
# mp3File = input("Enter a mp3 filename: ")
# Play the mp3 file
# playsound('alarm.mp3')