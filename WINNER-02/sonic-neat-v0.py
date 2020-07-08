import retro
import numpy as np
import cv2
import neat
import pickle

# create game environment
env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

# generation counter
gen_count = 14

# funtion to evaluate genomes during training process
def eval_genomes(genomes, config):

    # generation counter
    global gen_count
    gen_count += 1

    for genome_id, genome in genomes:

        # get environment print screen
        observation = env.reset()

        # set shape size to input in neural network
        inx, iny, _ = env.observation_space.shape
        inx = int(inx/8)
        iny = int(iny/8)

        # create the neural network
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # initialize variables
        max_fitness_current = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xposition = 0
        xposition_max = 0
        xposition_end = 0
        abef = 0
        acur = 0

        imgarray = []     

        done = False
        trying_back = False
        counter_back = 0
        xposition_max_back = 100000

        #######################################################################################################
        ###########################################control variables###########################################
        #######################################################################################################

        activate_position_reward = True # activate reward by moving forward in x
        position_reward = 0.1 # how much reward by moving forward in x
                    
        frame_speed_check = 0 # number of frames to check speed in x, set to 0 to deactivate
        speed_check = 10 # units moved in x

        frame_not_reward = 250 # number of frames without any reward, set done = True
        assert frame_not_reward > 0 # this variable can't be 0 or less!!!

        incentive_back = 250 # number of frames to try get back to get out of local minimum, try to find global minimum, set to 0 to deactivate
        reward_back = 0.05 # reward for get back

        log = True
        log_size = 100

        #######################################################################################################
        ###########################################control variables###########################################
        #######################################################################################################

        # create a window to show the gameplay
        cv2.namedWindow('SonicTheHedgehog-Genesis | NEAT-Python | jubatistim', cv2.WINDOW_NORMAL)
        cv2.moveWindow("SonicTheHedgehog-Genesis | NEAT-Python | jubatistim", 950, 120)
        cv2.resizeWindow('SonicTheHedgehog-Genesis | NEAT-Python | jubatistim', 800,600)

        # main loop
        while not done:

            # count frames
            frame += 1

            # show gameplay
            shwimg = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
            cv2.imshow('SonicTheHedgehog-Genesis | NEAT-Python | jubatistim', shwimg)
            cv2.waitKey(1)

            # prepare the print screen to use as neural network input
            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))

            for x in observation:
                for y in x:
                    imgarray.append(y)

            # process the print screen through the neural network to obtain the output (actions)
            nnOutput = net.activate(imgarray)

            # apply actions to game environment to get new observation (print screen), reward gain by applying the action, and current values of info parameters (data.json)
            observation, reward, done, info = env.step(nnOutput)

            # clear the input for the next loop
            imgarray.clear()

            # reward from scenario.json
            fitness_current += reward

            # get positions of the screen and lives
            xposition = info['x']
            xposition_end = info['screen_x_end']
            died = info['lives'] == 2

            # set current fitness to maximum when reach the end of the screen
            if xposition >= (xposition_end - 500) and xposition > 500:
                fitness_current = 100000
                done = True

            # give a little more reward when advance to a position higher than the maximum achieved
            # it's not possible to use the reward in scenario.json because it increment the position even if it's not the highest
            if xposition > xposition_max and activate_position_reward:
                fitness_current += position_reward
                xposition_max = xposition

            # speed, if it's slow stop
            if frame_speed_check > 0:
                if frame % frame_speed_check == 0:
                    abef = acur
                    acur = fitness_current

            # check progress, if it's not progressing stop
            if fitness_current > max_fitness_current and not trying_back:
                max_fitness_current = fitness_current
                counter = 0
            else:
                counter += 1            

            # incentive to get back a little to run out of local minimuns
            if counter >= frame_not_reward and incentive_back > 0:
                trying_back = True

                if xposition < xposition_max_back:
                    fitness_current += reward_back
                    xposition_max_back = xposition

                if fitness_current > max_fitness_current:
                    max_fitness_current = fitness_current
                
                counter_back += 1

                if counter_back >= incentive_back and xposition > xposition_max:
                    counter = 0
                    counter_back = 0
                    trying_back = False
                
            # prints for log
            if frame % log_size == 0 and log:
                print('gen_count: ', gen_count, 'frame: ', frame, 'fitness_current: ', fitness_current, 'max_fitness_current: ', max_fitness_current, 'counter: ', counter, 'speed_stop: ', (acur - abef), 'Trying back', counter_back,)

            # stop if not progressing
            stop_reason = ''
            if (incentive_back > 0 and counter_back >= incentive_back) or (incentive_back == 0 and counter >= frame_not_reward):
                stop_reason = 'Reached maximum frames without reward'
                done = True

            if (frame > frame_speed_check and (acur - abef) < speed_check and frame_speed_check > 0):
                stop_reason = 'Too slow'
                done = True

            if died:
                stop_reason = 'Died'
                done = True

            # log end
            if done and log:
                print('------------------------------------------------------------------------------------------------------------------')
                print('genome_id: ', genome_id, 'fitness_current: ', fitness_current, 'Stop reason: ', stop_reason)
                print('------------------------------------------------------------------------------------------------------------------')                

            # set genome fitness to train the neural network
            genome.fitness = fitness_current

# load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-sonic-neat-v0')

# set configuration
p = neat.Population(config)
# p = neat.Checkpointer.restore_checkpoint('./bkpchkpointers/neat-checkpoint-15')

# report trainning
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
# p.add_reporter(neat.Checkpointer(10)) #every x generations save a checkpoint

# run trainning
winner = p.run(eval_genomes)

# save the winner
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)