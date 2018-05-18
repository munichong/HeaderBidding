import pickle, matplotlib
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pprint import pprint
from collections import Counter
from survival_analysis.DataReader import SurvivalData

time_train, event_train, features_train = pickle.load(open('../Vectors_train.p', 'rb'))

def floor_distribution():
    time_counter = Counter()
    for t in time_train:
        time_counter[t] += 1

    print("MOST COMMON:", time_counter.most_common(10))
    for time, count in sorted(time_counter.items()):
        print(time, '-->', count)

def plot_event_vs_floor():
    floor0 = []
    floor1 = []
    for t, e in zip(time_train, event_train):
        if e == 0:
            floor0.append(t)
        else:
            floor1.append(t)

    pyplot.hist([floor0, floor1], color=['g','r'], bins=3000, label=['adx-won', 'adx-lose'])
    # pyplot.hist(labels1, bins, alpha=0.5, label='adx-lose')
    pyplot.legend(loc='upper right')
    pyplot.xlim(0, 10)
    pyplot.xticks([n / 10 for n in range(0, 100, 5)])
    pyplot.show()

def event_vs_floor():
    time_counter0 = Counter()
    time_counter1 = Counter()
    for t, e in zip(time_train, event_train):
        if e == 0:
            time_counter0[t] += 1
        else:
            time_counter1[t] += 1

    print("MOST COMMON 0 (adx-won):")
    pprint(time_counter0.most_common(10))
    print()
    print("MOST COMMON 1 (adx-lose):")
    pprint(time_counter1.most_common(10))

    print()
    print("SORTED ADX-WON FLOOR PRICES:")
    for time, count in sorted(time_counter0.items())[:50]:
        print(time, '-->', count)
    print("......")

    print()
    print("SORTED ADX_LOSE FLOOR PRICES:")
    for time, count in sorted(time_counter1.items())[:50]:
        print(time, '-->', count)
    print("......")

if  __name__ == '__main__':
    plot_event_vs_floor()