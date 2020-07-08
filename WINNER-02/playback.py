import retro
import numpy as np
import cv2 
import neat
import pickle

# create game environment
env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

# load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-sonic-neat-v0')

# load previous saved winner genome
with open('winner-02.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

# get environment first print screen
observation = env.reset()

# get image size
shpx, shpy, _ = env.observation_space.shape

# resize parameters
shpx = int(shpx/8)
shpy = int(shpy/8)

# create neural network with loaded genome
net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

# initialize variables
xposition = 0
xposition_max = 0

imgarray = []

done = False

# create a window to show the gameplay
cv2.namedWindow('SonicTheHedgehog-Genesis | NEAT-Python | jubatistim', cv2.WINDOW_NORMAL)
cv2.moveWindow("SonicTheHedgehog-Genesis | NEAT-Python | jubatistim", 950, 100)
cv2.resizeWindow('SonicTheHedgehog-Genesis | NEAT-Python | jubatistim', 800,600)

# main loop
while not done:

    # show gameplay
    shwimg = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
    cv2.imshow('SonicTheHedgehog-Genesis | NEAT-Python | jubatistim', shwimg)
    cv2.waitKey(1)

    # prepare the print screen to use as neural network input
    observation = cv2.resize(observation, (shpx, shpy))
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = np.reshape(observation, (shpx,shpy))

    imgarray = np.ndarray.flatten(observation)
    
    # process the print screen through the neural network to obtain the output (actions)
    nnOutput = net.activate(imgarray)
    
     # apply actions to game environment to get new observation (print screen), reward gain by applying the action, and current values of info parameters (data.json)
    observation, reward, done, info = env.step(nnOutput)
    
    # check the act number so set done
    act = info['act']

    # set done
    if act != 0:
        done = True
    