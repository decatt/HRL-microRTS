import matplotlib.pyplot as plt
import numpy as np


size = 6
worker = [0.0444160466255031,0.0669926932310176,0.053398590056873474,0.026979221346167155,0.029961641185716013,0.060328281650734435]
light = [0.00011943007965986314,0.0008648614649180741,0.00018921782756309063,6.388517704179687e-05,0.00235208775265709,4.2391742088641135e-05]
ranges = [0.0001672021115238084,0.002164774454613119,0.00017840538027377116,8.518023605572917e-05,4.396425705901102e-05,9.008245193836241e-05]
heavy = [0.0001672021115238084,0.0017978635301024205,0.0019462405120775036,0.0005004338868274089,6.594638558851654e-05,0.005823565569427076]
resource = [0.003881477588945552,0.0070656560891488715,0.00484397638561512,0.004972396279753191,0.0051218359473747845,0.007068822993280909]
attack = [0.0007046374699931925,1.0483169271734231e-05,1.0812447289319465e-05,0.0012883510703429037,7.144191772089291e-05,0.0009431951401470745]
# move = [0.25098231240520236,0.30005975406484886,0.2796747615855373,0.2631909581179427,0.24838156578701515]

name = ["random","reward: build worker","reward: resource","reward: attack","reward: build barracks units","reward: normal"]
x = np.arange(size)
total_width, n = 0.8, 6     # 有多少个类型，只需更改n即可
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, worker,  width=width, label='build worker',color='blue')
plt.bar(x + width, light, width=width, label='build light',color='green')
plt.bar(x + 2 * width, ranges, width=width, label='build range', color='yellow')
plt.bar(x + 3 * width, heavy, width=width, label='build heavy', color='gray', tick_label=name)
plt.bar(x + 4 * width, resource, width=width, label='return resource', color='cyan')
plt.bar(x + 5 * width, attack, width=width, label='attack', color='deepskyblue')
# plt.bar(x + 6 * width, move, width=width, label='move', color='red')


plt.xticks()
plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.ylabel('value')
plt.xlabel('type')
plt.title("actions of agent trained with different rewards")
plt.show()
