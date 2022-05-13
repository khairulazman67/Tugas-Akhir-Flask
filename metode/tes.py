import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# opsi angka 0 atau 1 menunjukkan bahwa kita mengambil data dari webcam
count=0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Opsi untuk menyimpan file dalam format gray
    
    nama_file = str(count) + '.jpg'
    cv2.imwrite(nama_file,frame)
    #Perintah untuk menyimpan file dengan nama yang berbeda

    count=count + 1
    # menambahkan count untuk mendapatkan nama file yang berbeda

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()