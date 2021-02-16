from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import pandas
import os

fig, (a1, a2, a3) = plt.subplots(3)

def animate(k):
    a1.cla()
    a2.cla()
    a3.cla()

    data = pandas.read_csv("results.csv")

    x = data['total_frames']
    y1 = data['final_reward']
    y2 = data['frames']
    y3 = data['avg_error']

    a1.plot(x, y1)
    a2.plot(x, y2)
    a3.plot(x, y3)

    a1.set_ylabel("Reward")
    a2.set_ylabel("Lifetime")
    a2.set_ylabel("TD-error")

    plt.xlabel("Frame")
    fig.suptitle("RL Statistics")

    plt.tight_layout()

anim = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()
