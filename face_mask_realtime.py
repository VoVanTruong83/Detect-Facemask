
from keras.models import Sequential,load_model
from keras.preprocessing import image
import cv2
import numpy as np
import datetime
import playsound
from threading import Thread

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("model_face_mask.h5")

i= "D:\warn.mp3"

def sound_alarm(path):
    playsound.playsound(path)


isWarnning = False
while True:
    _, img = cap.read()
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in face:
        face_img = img[y:y + h, x:x + w]
        cv2.imwrite('temp.jpg', face_img)
        test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        pred = model.predict_classes(test_image)[0][0]
        print(model.predict_classes(test_image)[0][0])
        if pred == 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, 'KHONG KHAU TRANG', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if isWarnning == False:
                t = Thread(target=sound_alarm,
                           args=('D:\warn.mp3',))
                
                t.start()
                isWarnning = True

            if t.is_alive() == False:
                isWarnning = False

        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'CO KHAU TRANG', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
     
        datet = str(datetime.datetime.now())
        cv2.putText(img, datet, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()