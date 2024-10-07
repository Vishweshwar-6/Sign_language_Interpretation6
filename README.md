# Sign Language Interpretation Using LSTM and MediaPipe

## Overview
  This project focuses on creating a Sign Language Interpreter using MediaPipe and Long Short-Term Memory (LSTM) neural networks. The primary goal is to help bridge the communication gap for deaf and hard-of-hearing individuals by translating American Sign Language (ASL) into English text in real-time.

  Sign language is a sophisticated visual language that involves complex hand, body, and facial gestures. This system utilizes deep learning techniques and computer vision to process video inputs of individuals performing sign language and translates it into readable English text.

## Objective
  The objective of this project is to develop a system that:

  Captures video of a person performing sign language.
  Extracts key features of the hand and arm movements using MediaPipe.
  Processes the data using an LSTM neural network to output corresponding English text.
  This system aims to improve accessibility for people who rely on sign language, allowing better interaction and communication with non-sign language speakers.

## Dataset
  We created our own dataset for this project, consisting of videos of signers performing various ASL signs. Each sign was recorded multiple times under varying lighting conditions, using both the left and right hands. The dataset was split into 80% training data and 20% testing data for model evaluation.

  The dataset contains:

  30 different frames for each sign.
  Different lighting settings and speeds to introduce variability.
  Right and left-hand recordings for one-handed signs (e.g., alphabets and numbers).

## Methodology
  #### Step 1: Pre-processing
    The raw video data is first pre-processed using MediaPipe. This step includes:
  
    Resizing and cropping video frames.
    Extracting hand, pose, and body landmarks using the MediaPipe Holistic Model.
    Storing the data points (in the form of numpy arrays) for use in the training process.
  #### Step 2: Feature Extraction
    Using MediaPipe, we capture:
    
    Hand landmarks (21 for each hand).
    Pose landmarks (33 for the body).
    This data is saved for training the LSTM model.

  #### Step 3: LSTM Neural Network
    We use LSTM, a type of Recurrent Neural Network (RNN) architecture, to model sequential data. The key features of this LSTM model are:
    
    Three LSTM layers followed by dense layers.
    Adam optimizer to minimize the loss function.
    The model was trained for 500 epochs.
    Achieved an accuracy of 95.4%.
    
  #### Step 4: Model Training
    The extracted features from the pre-processing step are used to train the LSTM model. We used TensorFlow as the primary deep learning library for model training.

  #### Step 5: Real-time Predictions
    The trained LSTM model is used to predict ASL gestures from video input. The system dynamically interprets the input and translates it into text on the screen.

## Project Architecture
  

## Tools and Technologies
    **Python**: Core programming language used.
    **TensorFlow**: Deep learning library for building the LSTM model.
    **MediaPipe**: For feature extraction and keypoint detection from video input.
    **OpenCV**: For handling image and video processing tasks.
    **NumPy**: For handling arrays and matrices.
    **Matplotlib**: For visualizing data and model performance.
    
## Results
  The system performs well with most signs being translated as expected. Some key observations:

  Correct Translations: Most ASL signs are translated accurately into English text.

  IMAGE
  
  Incorrect Translations: Occasional errors occur when signs are visually similar (e.g., "B" being interpreted as "4").

## Model Performance:
  Training Accuracy: 95.4%
  Training Epochs: 500
  Adam Optimizer: Used to minimize loss functions.
## Challenges and Future Work
  ### Challenges Faced:
    Data Quality: Low-quality videos can affect the model's accuracy.
    Sign Variability: Different individuals may perform signs differently, which can lead to misinterpretations.
    Processing Speed: Real-time interpretation requires fast processing to keep up with signer's movements.
