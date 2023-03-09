from helper_functions import *
import tensorflow as tf
import mediapipe as mp
import time
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model,save_model

# Data path
path = os.path.join('DATA')

# Array of each action to be detected
actions = np.array(['hello','welcome','thank you','goodbye'])

# How many videos will be collected for each action
no_sequences = 30

# How many frames each video in no_sequences will be
sequence_length = 30

# Creates the folders for each action and inside each action will be 30 folder each representing a video
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(path,action,str(sequence)))
        except:
            pass

# Initialize video capture
cap = cv2.VideoCapture(0)

# Detection model
with mp_hollistic.Holistic(min_tracking_confidence = 0.5, min_detection_confidence = 0.5) as holistic:
    '''
    for each action this block will iterate through actions array and collect 30 videos each 30 
    frames in length. Making detections and extracting the keypoint's frame by frame and saving them to the 
    corresponding frame in the corresponding video folder.
    '''
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret,frame = cap.read()
                image,results = mediapipe_detections(frame,holistic)
                draw_landmarks(image,results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                keypoint = extract_keypoints(results)
                npy_path = os.path.join(path,action,str(sequence),str(frame_num))
                np.save(npy_path,keypoint)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Process data into features and labels
label_map = {label:num for num,label in enumerate(actions)}
sequences,labels = [],[]
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(path, action))).astype(int):
        # Represents every frame in said sequence(could also be referred to as a video)
        window = []
        for frame_num in range(sequence_length):
            # Load each frame and append it to window
            res = np.load(os.path.join(path, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        # Append video to sequence list which represents X
        sequences.append(window)
        # Append corresponding label of said video to labels which represents Y
        labels.append(label_map[action])

X = np.array(sequences)
'''
One hot encodes labels for example if you have three actions it will recognize each array 
as an array of one 1 and two zeroes such as [0,0,1]
'''
y = to_categorical(labels).astype(int)

# Split data into trainng and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=1)

# Build and train model using helper functions then load trained model
model = build_model('Adam')
trained_model = train_model(model,X_train,y_train)
trained_model = tf.keras.models.load_model('trained_model')

# Make test predictions
res = model.predict(X_test)
print(actions[np.argmax(res[0])])
print(actions[np.argmax(y_test[0])])

colors = [(255, 255, 255), (245, 245,245), (16, 117, 245),(16, 117, 245)]


def prob_viz(res , actions_list, input_frame, colors_list):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame

# New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.4

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_hollistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detections(frame, holistic)
        print(results)

        # Draw landmarks
        draw_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = trained_model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

        # Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()