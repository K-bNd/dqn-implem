"""
Hyperparameters values
"""
RANDOM_SEED = 42
MINIBATCH_SIZE = 32
REPLAY_SIZE = 1e5
UPDATE_FREQUENCY = 10000
GAMMA = 0.99
RMS_LEARNING_RATE = 2.5e-4
RMS_GRADIENT_MOMENTUM = 0.95
RMS_ESP = 0.01
TAU = 0.005
MAX_EPOCHS = 1000
MAX_ACTION = 3600
# 6 (action per second) * 600 (seconds per game, 10mn of game max) = 3600 actions per game
