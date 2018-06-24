# A simple example of simulated annealing. 
# We're trying to optimize a random walk. Lower values are considered better. 
# After each iteration, it prints the score. It should hopefully drop quite a bit.

import numpy as np

old = 500
new = old
temperature = 1.0
count = 0
alpha = 0.9995
iteration_num = 10000
np.random.seed(555)


# Main while loop
while count < iteration_num:
    temperature *= alpha
    count += 1

    rand_delta = np.random.randint(-5, 15)
    new = old + rand_delta

    if new < old:
        old = new

    else:
        uniform_rand_num = np.random.uniform(0,1)
        p = np.exp(-((new - old)/temperature))

        if p > uniform_rand_num:
            old = new

    print "Temperature: {0} \t Score: {1}".format(temperature, old)