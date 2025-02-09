import pickle
import random

from matplotlib import pyplot as plt

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

objects = [
['scissors', 'handle', 'tip', 'blade'],
['extra large clamp', 'handle'],
['bowl', 'lip'],
['mug', 'handle', 'lip'],
['apple'],
['tennis ball'],
['pear'],
['lemon'],
['racquetball'],
['baseball'],
['cup', 'lip'],
['cup', 'lip'],
['cup', 'lip'],
['strawberry'],
['colored wood block'],
['colored wood block'],
['colored wood block'],
['phillips screwdriver', 'tip', 'handle'],
['flat screwdriver', 'tip', 'handle'],
['orange'],
['banana'],
['rubik\'s cube'],
['bleach cleanser', 'tip'],
['master chef can', 'lip'],
['potted meat can', 'tip'],
['tuna fish can', 'lip'],
['tomato soup can', 'lip'],
['chips can'],
['mustard bottle', 'tip']
]

def random_poses():
    def rand_pose():
        x_width, y_width, padding = .91, .91, .1
        return (random.random() * (x_width - 2 * padding) - (x_width - 2 * padding) / 2, random.random() * (y_width - 2 * padding) + padding)

    def touches(r):
        for x, y in poses:
            if abs(x - r[0]) < 0.07 and abs(y - r[1]) < 0.17: # 20 cm
                return True
        return False
    
    # Num Objects
    n, poses = 6, []
    for i in range(n):
        r = rand_pose()
        while touches(r):
            r = rand_pose()
        poses.append(r)

    object_parts = random.sample(objects, n)
    positions = [(a[0], b) for a, b in zip(object_parts, poses)]

    # Num queries
    queries = []
    for o in random.sample(object_parts, 3):
        choice = random.choice(o)
        if choice == o[0]:
            queries.append(f'Give me the {choice}')
        else:
            if random.random() < 0.5:
                queries.append(f'Give me the {o[0]} and grab the {choice}')
            else:
                queries.append(f'Give me the {o[0]} and let me hold the {choice}')

    return positions, queries, object_parts 

def gen():
    num_scenes, data = 50, []
    for _ in range(num_scenes):
        data.append(random_poses())

    with open('dataset/dataset.pickle', 'wb') as file:
        pickle.dump(data, file)

def view():
    with open('dataset/dataset.pickle', 'rb') as file:
        data = pickle.load(file)

    for index, entry in enumerate(data):
        print(f'Queries for scene {index}', entry[1])

    for index, entry in enumerate(data):
        print(f'Queries for scene {index}', entry[1])

        for name, (x, y) in entry[0]:
            plt.scatter(x, y, s=2000) # Plot each point
            plt.text(x, y, name, rotation=30, fontsize=10) # Annotate each point with its name

        plt.xlim(-0.6, 0.6)
        plt.ylim(-0.1, 1.1)
        plt.gca().set_aspect('equal', 'box')
        plt.gca().axvline(-0.5)
        plt.gca().axvline(0, linestyle='--')
        plt.gca().axvline(0.5)

        plt.gca().axhline(0)
        plt.gca().axhline(0.5, linestyle='--')
        plt.gca().axhline(1.0)
        plt.gcf().set_size_inches(9, 9)
        plt.show()

if __name__ == '__main__':
    print("started!")
    # gen()
    view()