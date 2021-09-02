import matplotlib.pyplot as plt
import numpy as np

record_path = './record/microrts_fun_0901_16.txt'
rewards_record=[]
outcomes_record=[]
with open(record_path,"r") as f:
    data = f.readlines()
    rewards_record = eval(data[0])
    outcomes_record = eval(data[2])

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.plot(np.array(range(len(rewards_record))), np.array(rewards_record), color='red')
    ax2.plot(np.array(range(len(outcomes_record))), np.array(outcomes_record), color='red')
    fig.canvas.draw()
    plt.show()
