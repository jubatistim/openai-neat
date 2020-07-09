import retro
import numpy as np
import cv2
import neat
import pickle

# create game environment
env = retro.make('SuperMarioWorld-Snes', 'DonutPlains1')

# create a window to show the gameplay
cv2.namedWindow('SuperMarioWorld-Snes | NEAT-Python | jubatistim', cv2.WINDOW_NORMAL)
cv2.moveWindow("SuperMarioWorld-Snes | NEAT-Python | jubatistim", 950, 120)
cv2.resizeWindow('SuperMarioWorld-Snes | NEAT-Python | jubatistim', 800,600)

# generation
generation = 9

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
        fitness_current_max = 0
        posx = 0
        posx_max = 0
        counter = 0
        frame = 0

        imgarray = []

        # main loop
        while not done:

            # frame count
            frame += 1
            
            # show gameplay
            shwimg = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
            cv2.imshow('SuperMarioWorld-Snes | NEAT-Python | jubatistim', shwimg)
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

            # get info
            lives = info['lives']
            posx = info['screen_x']

            # reward from scenario.json
            fitness_current += reward

            print('posx: ', posx, 'posx_max: ', posx_max, 'fitness_current: ', fitness_current)

            # reward for position
            if posx > posx_max:
                posx_max = posx
                fitness_current += 0.1

            # set counter to stop
            if fitness_current > fitness_current_max:
                fitness_current_max = fitness_current
                counter = 0
            else:
                counter += 1

            # set genome fitness to train the neural network
            genome.fitness = fitness_current

            # counter to stop
            reason_stoped = ''
            if counter > 1000:
                reason_stoped = 'Maximum frames without reward'
                done = True

            # if die stop
            if lives <= 3:
                reason_stoped = 'Died'
                done = True

            # logs
            if log and frame % log_size == 0:
                print('generation: ', generation, 'genome_id: ', genome_id, 'frame: ', frame, 'counter', counter, 'fitness_current: ', fitness_current)
            if done and log:
                print('------------------------------------------------------------------------------------------------------------------')
                print('generation: ', generation, 'genome_id: ', genome_id, 'frame: ', frame, 'counter', counter, 'fitness_current: ', fitness_current, 'Reason: ', reason_stoped)
                print('------------------------------------------------------------------------------------------------------------------')

# load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-super-mario-neat-v0')

# set configuration
p = neat.Population(config)
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-10')

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