import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    t = np.arange(60)
    g = skewnorm.pdf(t, 1, loc=28, scale=7)
    w = g / g.sum()
    u = np.ones((60,)) / 60
    p = 0.3
    w = p * u + w * (1 - p)

    sns.lineplot(w)
    print(w)
    plt.savefig('./time_scheduler_plot.jpg')


if __name__ == '__main__':
    main()