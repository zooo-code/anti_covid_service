import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
import time

model = load_model('mask_model.h5')
# model.summary()
no_stable = 0
yes_stable = 0

# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
ok_mask = []
no_mask = []
# loop through frames
pre_time = time.time()
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()
    # cv2.imshow("mask nomask frame", frame)
    if not status:
        print("Could not read frame")

        exit()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # print("face",face)
    # print("confidence",confidence)

    if face == []:
        print("no people")
        cv2.putText(frame, "no person...", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # loop through detected faces
    for idx, f in enumerate(face):
        print(face)
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[
            0] and 0 <= endY <= frame.shape[0]:

            face_region = frame[startY:endY, startX:endX]

            face_region1 = cv2.resize(face_region, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imshow("mask nomask face_region1", face_region1)
            x = img_to_array(face_region1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            prediction = model.predict(x)
            print(prediction)

            if prediction < 0.7:  # 마스크 미착용으로 판별되면,
                no_mask.append("N")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "No Mask ({:.2f}%)".format((1 - prediction[0][0]) * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(frame, "no mask!", (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:  # 마스크 착용으로 판별되면
                ok_mask.append("Y")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Mask ({:.2f}%)".format(prediction[0][0] * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # print("ok_mask", ok_mask)
            #
            # cur_time=time.time()
            # time_gap =cur_time - pre_time
            # print("time_gap",time_gap)
            # if pre_time +10 <= cur_time:
            #     pre_time=time.time()
            #     print("ok_mask",ok_mask)
            #     print("ok_mask_len",len(ok_mask))
            #     ok_mask = []
            #     no_mask = []

    # display output
    cv2.imshow("mask nomask classify", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows() 