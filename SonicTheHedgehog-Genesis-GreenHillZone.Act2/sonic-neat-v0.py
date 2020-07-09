import retro
import numpy as np
import cv2
import neat
import pickle

# create game environment
env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act2')
# env = retro.make('SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act1')

# generation counter
gen_count = 0

# funtion to evaluate genomes during training process
def eval_genomes(genomes, config):

    # generation counter
    global gen_count
    gen_count += 1

    for genome_id, genome in genomes:

        # get environment print screen
        ob = env.reset()

        # set shape size to input in neural network
        inx, iny, _ = env.observation_space.shape
        inx = int(inx/8)
        iny = int(iny/8)

        # create the neural network
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # initialize variables
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0
        xpos_end = 0
        abef = 0
        acur = 0

        imgarray = []     

        done = False
        trying_back = False
        counter_back = 0
        xpos_max_back = 100000

        #######################################################################################################
        ###########################################control variables###########################################
        #######################################################################################################

        activate_position_reward = True # activate reward by moving forward in x
        position_reward = 0.1 # how much reward by moving forward in x
                    
        frame_speed_check = 0 # number of frames to check speed in x, set to 0 to deactivate
        speed_check = 10 # units moved in x

        frame_not_reward = 250 # number of frames without any reward, set done = True
        assert frame_not_reward > 0 # this variable can't be 0 or less!!!

        log = True
        log_size = 40

        #######################################################################################################
        ###########################################control variables###########################################
        #######################################################################################################

        # create a window to show the gameplay
        cv2.namedWindow('SonicTheHedgehog-Genesis and NEAT-Python', cv2.WINDOW_NORMAL)
        cv2.moveWindow("SonicTheHedgehog-Genesis and NEAT-Python", 950, 100)
        cv2.resizeWindow('SonicTheHedgehog-Genesis and NEAT-Python', 800,600)

        # main loop
        while not done:

            # count frames
            frame += 1

            # show gameplay
            shwimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            cv2.imshow('SonicTheHedgehog-Genesis and NEAT-Python', shwimg)
            cv2.waitKey(1)

            # prepare the print screen to use as neural network input
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            for x in ob:
                for y in x:
                    imgarray.append(y)

            # process the print screen through the neural network to obtain the output (actions)
            nnOutput = net.activate(imgarray)

            # apply actions to game environment to get observation (print screen), reward gain by applying the action, and current values of info parameters (data.json)
            ob, rew, done, info = env.step(nnOutput)

            # clear the input for the next loop
            imgarray.clear()

            # reward from scenario.json
            fitness_current += rew

            # get positions of the screen
            xpos = info['x']
            xpos_end = info['screen_x_end']

            # set current fitness to maximum when reach the end of the screen
            if xpos >= (xpos_end - 500) and xpos > 500:
                fitness_current = 100000
                done = True

            # give a little more reward when advance to a position higher than the maximum achieved
            # it's not possible to use the reward in scenario.json because it increment the position even if it's not the highest
            if xpos > xpos_max and activate_position_reward:
                fitness_current += position_reward
                xpos_max = xpos

            # speed, if it's slow stop
            if frame_speed_check > 0:
                if frame % frame_speed_check == 0:
                    abef = acur
                    acur = fitness_current

            # check progress, if it's not progressing stop
            if fitness_current > current_max_fitness and not trying_back:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            # prints for log
            if frame % log_size == 0 and log:
                #print('xpos: ', xpos, 'xpos_max: ', xpos_max, 'xpos_end: ', xpos_end)
                print('gen_count: ', gen_count, 'frame: ', frame, 'fitness_current: ', fitness_current, 'current_max_fitness: ', current_max_fitness, 'counter: ', counter, 'speed_stop: ', (acur - abef))

          
            if counter >= frame_not_reward:
                trying_back = True

                if xpos < xpos_max_back:
                    fitness_current += 0.05
                    xpos_max_back = xpos

                if fitness_current > current_max_fitness:
                    current_max_fitness = fitness_current
                
                counter_back += 1

                if counter_back >= 250 and xpos > xpos_max:
                    counter = 0
                    counter_back = 0
                    trying_back = False

                print('Trying back', counter_back, 'fitness_current', fitness_current)
                
                

            # stop if not progressing
            stop_reason = ''
            if counter_back >= 250:
                stop_reason = 'Reached maximum frames without reward'
                done = True

            if (frame > frame_speed_check and (acur - abef) < speed_check and frame_speed_check > 0):
                stop_reason = 'Too slow'
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
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-17')

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