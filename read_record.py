import matplotlib.pyplot as plt
import numpy as np
# './record/microrts_ppo_2021092207_16.txt'

# microrts_GNNppo_2021111016_10.txt
record_path = './record/microrts_GNNppo_2021111319_10.txt'
record_path2 = './record/microrts_ppo_2021111415_10.txt'
record_path3 = './record/microrts_GNNppo_2021111707_10.txt'
record_path4 = './record/microrts_GNN_ppo_2021111920_10.txt'
record_path5 = './record/microrts_GNN_ppo_2021112103_10.txt'
#'./record/microrts_GNN_ppo_2021112310.txt'
rewards_record=[]
outcomes_record=[]
with open(record_path,"r") as f:
    data = f.readlines()
    rewards_record = eval(data[0])
    outcomes_record = eval(data[2])

rewards_record2=[]
outcomes_record2=[]
with open(record_path2,"r") as f:
    data = f.readlines()
    rewards_record2 = eval(data[0])
    outcomes_record2 = eval(data[2])

rewards_record3=[]
outcomes_record3=[]
with open(record_path3,"r") as f:
    data = f.readlines()
    rewards_record3 = eval(data[0])
    outcomes_record3 = eval(data[2])

rewards_record4=[]
outcomes_record4=[]
with open(record_path4,"r") as f:
    data = f.readlines()
    rewards_record4 = eval(data[0])
    outcomes_record4 = eval(data[2])

rewards_record5=[]
outcomes_record5=[]
with open(record_path5,"r") as f:
    data = f.readlines()
    rewards_record5 = eval(data[0])
    outcomes_record5 = eval(data[2])

fig = plt.figure()
plt.subplot(121)
plt.title("different method in 10x10 map")
plt.plot(np.array(range(len(rewards_record))), np.array(rewards_record), color='red', label='GNN')
plt.plot(np.array(range(len(rewards_record2))), np.array(rewards_record2), color='blue', label='CNN')
plt.plot(np.array(range(len(rewards_record3))), np.array(rewards_record3), color='green', label='supGNN (pre-trained)')
plt.plot(np.array(range(len(rewards_record4))), np.array(rewards_record4), color='orange', label='directionGNN')
plt.plot(np.array(range(len(rewards_record5))), np.array(rewards_record5), color='black', label='directionGNN(against multi AI)')
plt.ylabel("average rewards")
plt.xlabel("time(20*1024step)")
plt.legend()
plt.subplot(122)
plt.plot(np.array(range(len(outcomes_record))), np.array(outcomes_record), color='red', label="GNN")
plt.plot(np.array(range(len(outcomes_record2))), np.array(outcomes_record2), color='blue', label="CNN")
plt.plot(np.array(range(len(outcomes_record3))), np.array(outcomes_record3), color='green', label="supGNN")
plt.plot(np.array(range(len(outcomes_record4))), np.array(outcomes_record4), color='orange', label='directionGNN')
plt.plot(np.array(range(len(outcomes_record5))), np.array(outcomes_record5), color='black', label='directionGNN(against multi AI)')
plt.ylabel("average outcomes")
plt.xlabel("time(20*1024step)")
plt.legend()
fig.canvas.draw()
plt.show()
