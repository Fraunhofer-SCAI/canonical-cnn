import random
import datetime
import subprocess
subprocess.call('pwd')

print('running mnist data over the different random seed')

jobs=[]

#for each_seed in seed_values:
#    time_str = str(datetime.datetime.today())
#    print(each_seed, 'at time: ', time_str)
#    with open ('b_'+str(each_seed)+time_str+'.txt', 'w') as f:
#        jobs.append(subprocess.Popen(['/home/lveeramacheneni/lconda_env/bin/python', 'mnist.py', '--lr=2e-5', '--epochs=100', '--mode=1', '--seed', str(each_seed)], stdout=f))

for index in range(1, 11):
    time_str = str(datetime.datetime.today())
    print('Experiment: ', index, ' at time: ', time_str)    
    with open ('b_'+time_str+'.txt', 'w') as f:
        jobs.append(subprocess.Popen(['/home/lveeramacheneni/lconda_env/bin/python', '/home/lveeramacheneni/network-compression/src/small_mnist/mnist.py', '--lr=1e-3', '--epochs=50', '--mode=1'], stdout=f))

for job in jobs:
    job.wait()