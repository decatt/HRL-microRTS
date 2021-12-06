import matplotlib.pyplot as plt
import numpy as np
# './record/microrts_ppo_2021092207_16.txt'

# microrts_GNNppo_2021111016_10.txt
record_path = './record/microrts_GNN_ppo_2021120522.txt'
#record_path2 = './record/microrts_GNN_ppo_2021113013.txt'
#'./record/microrts_GNN_ppo_2021112310.txt'
rewards_record=[]
outcomes_record=[]
rewards_record2=[]
outcomes_record2=[]
with open(record_path,"r") as f:
    data = f.readlines()
    rewards_record = eval(data[0])
    outcomes_record = eval(data[2])
"""
with open(record_path2,"r") as f:
    data = f.readlines()
    rewards_record2 = eval(data[0])
    outcomes_record2 = eval(data[2])
"""
fig = plt.figure()
plt.subplot(121)
plt.title("different method in 10x10 map vs Coach AI")
plt.plot(np.array(range(len(rewards_record))), np.array(rewards_record), color='red', label='GNN')
#plt.plot(np.array(range(len(rewards_record2))), np.array(rewards_record2), color='blue', label='CNN')

plt.ylabel("average rewards")
plt.xlabel("time(20*1024step)")
plt.legend()
plt.subplot(122)
plt.plot(np.array(range(len(outcomes_record))), np.array(outcomes_record), color='red', label="GNN")
#plt.plot(np.array(range(len(outcomes_record2))), np.array(outcomes_record2), color='blue', label="CNN")

plt.xlabel("time(20*1024step)")
plt.legend()
fig.canvas.draw()
plt.show()