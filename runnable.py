import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from pyvirtualdisplay import Display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import pickle

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def ReLU(x):
    x[x<0] = 0
    return x

def forward(model, x):
    
    # (Hx1) = (HxD) * (Dx1)
    hidden_layer_logits = model["W1"] @ x 

    # ReLU(logits)
    hidden_layer_activation = ReLU(hidden_layer_logits)

    # (1x1) = (Hx1).T * (Hx1)
    output_layer_logits = model["W2"] @ hidden_layer_activation

    # Scalar with probability of going up
    up_prob = sigmoid(output_layer_logits)

    return up_prob


prev_x = None

def update_input(prev_x, cur_x, D):
    if prev_x is not None:
        x = cur_x - prev_x
    else:
        x = np.zeros(D)
    return x

def frame_preprocessing(observation_frame):
    # Crop the frame.
    observation_frame = observation_frame[35:195]
    # Downsample the frame by a factor of 2.
    observation_frame = observation_frame[::2, ::2, 0]
    # Remove the background and apply other enhancements.
    observation_frame[observation_frame == 144] = 0  # Erase the background (type 1).
    observation_frame[observation_frame == 109] = 0  # Erase the background (type 2).
    observation_frame[observation_frame != 0] = 1  # Set the items (rackets, ball) to 1.
    # Return the preprocessed frame as a 1D floating-point array.
    return np.array(observation_frame.astype(float))

model = np.load("model_checkpoint.pkl",allow_pickle=True)

env = gym.make("ALE/Pong-v5", render_mode="human")
observation, _ = env.reset()
# rng = np.random.default_rng(seed=12288743)

for i in range(10000):

    
    
    cur_x = frame_preprocessing(observation).ravel()
    x = update_input(prev_x, cur_x, 80*80)
    prev_x = cur_x

    prob = forward(model,x)

    action = 0
    if prob > 0.7:
        # Action 2 refers to moving up
        action = 2
    elif prob < 0.3:
        # Action 3 refers to moving down
        action = 3
    else:
        action = 1

    observation, reward, done, *_ = env.step(action)
    
    env.render()

    