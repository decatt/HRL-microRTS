import matplotlib.pyplot as plt
import numpy as np
# './record/microrts_ppo_2021092207_16.txt'
record_path = './record/microrts_fun_2021101307_10.txt'
record_path2 = './record/microrts_ppo_2021092209_16.txt'
record_path3 = './record/microrts_ppo_2021090913_16.txt'
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

fig = plt.figure()
plt.subplot(121)
plt.title("different police in 16x16 map vs Coach AI")
plt.plot(np.array(range(len(rewards_record))), np.array(rewards_record), color='red', label='produce heavy')
#plt.plot(np.array(range(len(rewards_record2))), np.array(rewards_record2), color='blue', label='produce light')
#plt.plot(np.array(range(len(rewards_record3))), np.array(rewards_record3), color='green', label='produce range')
plt.ylabel("average rewards")
plt.xlabel("time(20*1024step)")
plt.legend()
plt.subplot(122)
plt.plot(np.array(range(len(outcomes_record))), np.array(outcomes_record), color='red', label="produce heavy")
#plt.plot(np.array(range(len(outcomes_record2))), np.array(outcomes_record2), color='blue', label="produce light")
#plt.plot(np.array(range(len(outcomes_record3))), np.array(outcomes_record3), color='green', label="produce range")
plt.ylabel("average outcomes")
plt.xlabel("time(20*1024step)")
plt.legend()
fig.canvas.draw()
plt.show()
