import retro
import numpy as np
import cv2
import neat
import pickle

# create game environment
env = retro.make('Columns-Genesis', 'Arcade.Easy.Level0')

# create a window to show the gameplay
cv2.namedWindow('Columns-Genesis | NEAT-Python | jubatistim', cv2.WINDOW_NORMAL)
cv2.moveWindow("Columns-Genesis | NEAT-Python | jubatistim", 950, 120)
cv2.resizeWindow('Columns-Genesis | NEAT-Python | jubatistim', 800,600)

# generation
generation = -1

# funtion to evaluate genomes during training process
def eval_genomes(genomes, config):

    # generation
    global generation
    generation += 1

    for genome_id, genome in genomes:

        #######################################################################################################
        ###########################################control variables###########################################
        #######################################################################################################

        log = True
        log_size = 300

        #######################################################################################################
        ###########################################control variables###########################################
        #######################################################################################################

        # get environment print screen
        observation = env.reset()

        # set shape size to input in neural network
        inx, iny, _ = env.observation_space.shape
        inx = int(inx/8) #28
        iny = int(iny/8) #40

        # create the neural network
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # initialize variables
        done = False
        fitness_current = 0
        frame = 0

        imgarray = []

        # main loop
        while not done:

            # frame count
            frame += 1
            
            # show gameplay
            shwimg = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
            cv2.imshow('Columns-Genesis | NEAT-Python | jubatistim', shwimg)
            cv2.waitKey(1)
            
            # prepare the print screen to use as neural network input
            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))

            imgarray = np.ndarray.flatten(observation)

            # process the print screen through the neural network to obtain the output (actions)
            nnOutput = net.activate(imgarray)

            # apply actions to game environment to get new observation (print screen), reward gain by applying the action, and current values of info parameters (data.json)
            observation, reward, done, info = env.step(nnOutput)

            # reward from scenario.json
            fitness_current += reward

            # set genome fitness to train the neural network
            genome.fitness = fitness_current

            # logs
            if log and frame % log_size == 0:
                print('generation: ', generation, 'genome_id: ', genome_id, 'frame: ', frame, 'fitness_current: ', fitness_current)
            if done and log:
                print('------------------------------------------------------------------------------------------------------------------')
                print('generation: ', generation, 'genome_id: ', genome_id, 'fitness_current: ', fitness_current)
                print('------------------------------------------------------------------------------------------------------------------')

# load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-columns-neat-v0')

# set configuration
p = neat.Population(config)
# p = neat.Checkpointer.restore_checkpoint('./bkpchkpointers/neat-checkpoint-15')

# report trainning
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10)) #every x generations save a checkpoint

# run trainning
winner = p.run(eval_genomes)

# save the winner
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

# close environment
env.close()