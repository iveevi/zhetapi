from matplotlib.animation import FuncAnimation

import json
import matplotlib.pyplot as plt
import os
import pandas

with open('config.json') as file:
    config = json.load(file)

files = {}

for strategy in config['Suites']:
    files[strategy['Name']] = strategy['Lib']

fig, (a1, a2, a3) = plt.subplots(3)

def animate(k):
    a1.cla()
    a2.cla()
    a3.cla()

    for kv in files:
        data = pandas.read_csv("res/" + files[kv] + ".csv")

        x = data['total_frames']
        y1 = data['final_reward']
        y2 = data['frames']
        y3 = data['avg_error']

        a1.plot(x, y1, label=kv)
        a2.plot(x, y2, label=kv)
        a3.plot(x, y3, label=kv)

    a1.set_ylabel("Reward")
    a2.set_ylabel("Lifetime")
    a3.set_ylabel("TD-error")

    a1.legend()
    a2.legend()
    a3.legend()

    plt.xlabel("Frame")
    fig.suptitle("RL Statistics")

    plt.tight_layout()

anim = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()
