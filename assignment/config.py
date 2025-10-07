DURATION = 10 # 10s sim
LAT_LAMBDA = 1 # lateral movement penalty
INPUT_SIZE = 13 # len(data.qpos) (15) - 3 head global positional args + sinusoidal clock
FIRST_HIDDEN_SIZE = 16 # custom, 'funnel' effect 
SECOND_HIDDEN_SIZE = 12
OUTPUT_SIZE = 8 # controls
#TILT = -5 # sloped world tilt along x axis


N = 20
pos = [0.0, 0, 0.2]
TARGET_POSITION = [5, 0, 0.5] 


# this can help understand the sizes - 1488
GENOME_SIZE = INPUT_SIZE*FIRST_HIDDEN_SIZE + \
                FIRST_HIDDEN_SIZE*SECOND_HIDDEN_SIZE + \
                SECOND_HIDDEN_SIZE*OUTPUT_SIZE