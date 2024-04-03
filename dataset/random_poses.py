import math
import pickle
import random

from matplotlib import pyplot as plt
from objects import objects

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


def random_poses():
    def rand_pose():
        return (random.random() * 1 - 0.5, random.random() * 0.5)
    
    poses = []

    def touches(r):
        for x, y in poses:
            if math.hypot(x - r[0], y - r[1]) < 0.15: # 20 cm
                return True
        return False
    
    # Num Objects
    n = 10

    for i in range(n):
        r = rand_pose()
        while touches(r):
            r = rand_pose()
        poses.append(r)

    object_parts = random.sample(objects, n)
    positions = [(a[0], b) for a, b in zip(object_parts, poses)]

    queries = []

    # Num queries
    for o in random.sample(object_parts, 2):
        choice = random.choice(o)
        if choice == o[0]:
            queries.append(f'Give me the {choice}')
        else:
            if random.random() < 0.5:
                queries.append(f'Give me the {o[0]} and grab the {choice}')
            else:
                queries.append(f'Give me the {o[0]} and let me hold the {choice}')

    # print(positions, '\n', queries, '\n')
    print(queries)

    return positions, queries

def gen():
    data = []
    for i in range(50):
        data.append(random_poses())

    with open('dataset/dataset.pickle', 'wb') as file:
        pickle.dump(data, file)

def view():
    with open('dataset/dataset.pickle', 'rb') as file:
        data = pickle.load(file)

    for entry in data:
        print(entry[1])

        for name, (x, y) in entry[0]:
            plt.scatter(x, y) # Plot each point
            plt.text(x, y, name) # Annotate each point with its name
        plt.xlim(-0.6, 0.6)
        plt.ylim(-0.1, 0.6)
        plt.gca().set_aspect('equal', 'box')
        plt.show()

if __name__ == '__main__':
    gen()
    view()