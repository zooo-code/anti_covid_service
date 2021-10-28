# Initialize the VideoCapture object to read from the webcam.

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils
# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

def getHandType(image, results, draw=True, display=True):

    # Create a copy of the input image to write hand type label on.
    output_image = image.copy()

    # Initialize a dictionary to store the classification info of both hands.
    hands_status = {'Right': False, 'Left': False, 'Right_index': None, 'Left_index': None}

    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):

        # Retrieve the label of the found hand.
        hand_type = hand_info.classification[0].label

        # Update the status of the found hand.
        hands_status[hand_type] = True

        # Update the index of the found hand.
        hands_status[hand_type + '_index'] = hand_index

        # Check if the hand type label is specified to be written.
        if draw:
            # Write the hand type on the output image.
            cv2.putText(output_image, hand_type + ' Hand Detected', (10, (hand_index + 1) * 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 2)

    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and the hands status dictionary that contains classification info.
        return output_image, hands_status

def detectHandsLandmarks(image, hands, display=True):

    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Check if landmarks are found.
    if results.multi_hand_landmarks:

        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS)

            # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        # plt.figure(figsize=[15, 15])
        # plt.subplot(121);
        # plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and results of hands landmarks detection.
        return output_image, results


def detectHandsLandmarks(image, hands, display=True):

    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Check if landmarks are found.
    if results.multi_hand_landmarks:
        pass
        # Iterate over the found hands.
        # for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the copy of the input image.
            # mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
            # Check if the original input image and the output image are specified to be displayed.
    if display:
        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and results of hands landmarks detection.
        return output_image, results



def drawBoundingBoxes(image, results, hand_status, padd_amount=10, draw=True, display=True):
    global x1_r , y1_r , x2_r, y2_r , x1_l,  y1_l,  x2_l,  y2_l
    # Create a copy of the input image to draw bounding boxes on and write hands types labels.
    output_image = image.copy()

    # Initialize a dictionary to store both (left and right) hands landmarks as different elements.
    output_landmarks = {}

    # Get the height and width of the input image.
    height, width, _ = image.shape
    two_hand = [0,0]
    # Iterate over the found hands.
    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

        # Initialize a list to store the detected landmarks of the hand.
        landmarks = []

        # Iterate over the detected landmarks of the hand.
        for landmark in hand_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

        # Get all the x-coordinate values from the found landmarks of the hand.
        x_coordinates = np.array(landmarks)[:, 0]

        # Get all the y-coordinate values from the found landmarks of the hand.
        y_coordinates = np.array(landmarks)[:, 1]

        # Get the bounding box coordinates for the hand with the specified padding.
        x1 = int(np.min(x_coordinates) - padd_amount)
        y1 = int(np.min(y_coordinates) - padd_amount)
        x2 = int(np.max(x_coordinates) + padd_amount)
        y2 = int(np.max(y_coordinates) + padd_amount)

        # Initialize a variable to store the label of the hand.
        label = "Unknown"
        # Check if the hand we are iterating upon is the right one.
        # 오른손 좌표 저장
        if hand_status['Right_index'] == hand_index:

            x1_r = int(np.min(x_coordinates) - padd_amount)
            y1_r = int(np.min(y_coordinates) - padd_amount)
            x2_r = int(np.max(x_coordinates) + padd_amount)
            y2_r = int(np.max(y_coordinates) + padd_amount)
            # print("r",x1_r,y1_r,x2_r,y2_r)
            # Update the label and store the landmarks of the hand in the dictionary.
            label = 'Right Hand'
            output_landmarks['Right'] = landmarks

            # 왼손 있다고 저장
            two_hand[0] = 1
        # Check if the hand we are iterating upon is the left one.
        # 왼손 좌표저장
        elif hand_status['Left_index'] == hand_index:

            x1_l = int(np.min(x_coordinates) - padd_amount)
            y1_l = int(np.min(y_coordinates) - padd_amount)
            x2_l = int(np.max(x_coordinates) + padd_amount)
            y2_l = int(np.max(y_coordinates) + padd_amount)


            # Update the label and store the landmarks of the hand in the dictionary.
            label = 'Left Hand'
            output_landmarks['Left'] = landmarks
            two_hand[1] = 1

        # Check if the bounding box and the classified label is specified to be written.
        if draw:
            # Draw the bounding box around the hand on the output image.
            # cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_8)
            pass
            # Write the classified label of the hand below the bounding box drawn.
            # cv2.putText(output_image, label, (x1, y2 + 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (20, 255, 155), 1,cv2.LINE_AA)


    # 손 바운딩 박스 그리기
    if two_hand[0]== 1 and two_hand[1]==1:
        if y1_r <= y1_l:
            two_hand_bbox_y1 = y1_r
        else:
            two_hand_bbox_y1 = y1_l

        if y2_r >= y2_l:
            two_hand_bbox_y2 = y2_r
        else:
            two_hand_bbox_y2 = y2_l

        two_hand_bbox_x1 = x1_l
        two_hand_bbox_x2 = x2_r

        # cv2.rectangle(output_image, (two_hand_bbox_x1, two_hand_bbox_y1), (two_hand_bbox_x2, two_hand_bbox_y2),(0, 255, 0), 3, cv2.LINE_8)
        hand_box = [[two_hand_bbox_x1,two_hand_bbox_y1],[two_hand_bbox_x2,two_hand_bbox_y2]]
    elif two_hand[0]== 1:
        one_hand_xr = x1_r # - (x2_r - x1_r)
        # cv2.rectangle(output_image, (one_hand_xr, y1_r), (x2_r, y2_r),(0, 255, 0), 3, cv2.LINE_8)
        hand_box = [[one_hand_xr, y1_r], [x2_r, y2_r]]
    elif two_hand[1] ==1:
        one_hand_xl = x2_l #  + (x2_l - x1_l)
        # cv2.rectangle(output_image, (x1_l, y1_l), (one_hand_xl, y2_l),(0, 255, 0), 3, cv2.LINE_8)
        hand_box = [[x1_l, y1_l], [one_hand_xl, y2_l]]
    else:
        pass
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and the landmarks dictionary.
        return output_image, output_landmarks ,hand_box


camera_video = cv2.VideoCapture("./video/s.MOV")
hands_video = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                             min_detection_confidence=0.7, min_tracking_confidence=0.4)
sample_num = 0
captured_num = 0

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()
    sample_num = sample_num + 1

    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Perform Hands landmarks detection.
    frame, results = detectHandsLandmarks(frame, hands_video, display=False)

    # Check if landmarks are found in the frame.
    if results.multi_hand_landmarks:
        # Perform hand(s) type (left or right) classification.
        _, hands_status = getHandType(frame.copy(), results, draw=False, display=False)

        # Draw bounding boxes around the detected hands and write their classified types near them.
        frame, _ , hand_box= drawBoundingBoxes(frame, results, hands_status, display=False)

        (startX, startY) = int(hand_box[0][0]*0.85), int(hand_box[0][1]*0.85)
        (endX, endY) = int(hand_box[1][0]*1.15),int(hand_box[1][1]*1.15)

        if sample_num % 2 == 0:

            captured_num = captured_num + 1
            try:
                hand_in_img = frame[startY:endY, startX:endX, :]

                cv2.imshow("hand_in_img",hand_in_img)
                cv2.imwrite('./hand1/hands_self' + str(captured_num) + '.jpg', hand_in_img)
            except:
                pass
    # Display the frame.
    cv2.imshow('Hands Landmarks Detection', frame)

    # Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()