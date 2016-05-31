import matplotlib.pyplot as plt
import numpy as np

def plot_mvn(mu,var,x=None,c='b',alpha=.2):
    if x is None:
        x = range(mu.shape[0])
    plt.plot(x,mu,c=c)
    plt.fill_between(x,mu-np.sqrt(var)*2,mu+np.sqrt(var)*2,color=c,alpha=alpha)
