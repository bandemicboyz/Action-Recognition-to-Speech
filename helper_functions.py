import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard

# Holistic model
mp_hollistic = mp.solutions.holistic

# Utensils for drawing
mp_drawing = mp.solutions.drawing_utils

# Array of each action you will be collection detections for
actions = np.array(['hello','welcome','thanks','goodbye'])

# Processes image and makes initial detections
def mediapipe_detections(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

# Visualize each landmark
def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_hollistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_hollistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_hollistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_hollistic.POSE_CONNECTIONS)

# Extract each set of keypoints
def extract_keypoints(results):
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])

# Describes the topology of the model and compile it
def build_model(opt:str):
    model = Sequential()
    model.add(LSTM(64,return_sequences = True,activation = 'relu', input_shape = (30,1662)))
    model.add(LSTM(128,return_sequences = True,activation = 'relu'))
    model.add(LSTM(64,return_sequences = False,activation = 'relu'))
    model.add(Dense(64,activation = 'relu'))
    model.add(Dense(32,activation = 'relu'))
    model.add(Dense(actions.shape[0],activation = 'softmax'))
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    return model


log_dir = 'log_dir'
tb_callback = TensorBoard(log_dir = log_dir)

def train_model(model,X_train,y_train,epochs = 500):
    model.fit(X_train,y_train,epochs = epochs ,callbacks = [tb_callback])
    model.save('trained_model')
    trained_model = tf.keras.models.load_model('trained_model')
    print(trained_model.summary())
    return trained_model

