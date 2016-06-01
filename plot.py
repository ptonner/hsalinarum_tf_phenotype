import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage, leaves_list

def plot_mvn(mu,var,x=None,c='b',alpha=.2):
    if x is None:
        x = range(mu.shape[0])
    plt.plot(x,mu,c=c)
    plt.fill_between(x,mu-np.sqrt(var)*2,mu+np.sqrt(var)*2,color=c,alpha=alpha)

def plot_delta(x,deltas,mean=True,probability=False,cluster=False,cluster_kwargs={}):
    p = len(deltas.keys())
    n = x.shape[0]
    a = np.zeros((p,n))
    yticks = deltas.keys()

    for i,k in enumerate(deltas.keys()):
        mu,var = deltas[k]

        if mean:
            a[i,:] = mu
        else:
            a[i,:] = 1-scipy.stats.norm.cdf(0,mu,np.sqrt(var))
            a[np.abs(a-.5) < .475] = 0.5

    if cluster:
        l = linkage(a,**cluster_kwargs)
        ind = leaves_list(l)
        a = a[ind,:]
        yticks = [yticks[j] for j in ind]

    if mean:
        lim = np.max(np.abs(a))
        vmin = -lim
        vmax = lim
    else:
        vmin = 0
        vmax = 1

    plt.imshow(a,cmap="RdBu",interpolation="none",vmin=vmin,vmax=vmax,origin='lower')
    plt.yticks(range(p),yticks)
    # i = plt.xticks()[0]
    i = np.arange(0,n,1.*n/5)
    plt.xticks(i,[x[j].round(2) for j in i])
    plt.colorbar()

def plot_model(x,gp,strain):
    time = np.linspace(0,43)

    plt.subplot(121)

    predx = patsy.build_design_matrices([x.design_info],{'time':time,'Strain':['ura3']*50})[0]
    mu,var = gp.predict(predx[:,1:])
    mu = mu[:,0]
    var = var[:,0]

    plt.plot(time,mu,color='k')
    plt.fill_between(time,mu-2*np.sqrt(var),mu+2*np.sqrt(var),color='k',alpha=.2)

    plt.subplot(122)
    predx = patsy.build_design_matrices([x.design_info],{'time':time,'Strain':[strain]*50})[0]
    mu,var = gp.predict(predx[:,1:])
    mu = mu[:,0]
    var = var[:,0]

    plt.plot(time,mu,color='g')
    plt.fill_between(time,mu-2*np.sqrt(var),mu+2*np.sqrt(var),alpha=.2,color='g')

def plot_data(data,strain):
    g = data.groupby("Strain")

    temp = g.get_group('ura3')
    g2 = temp.groupby(['Experiment','Well'])

    ylim = (np.round(temp.OD.min() - .5,1),np.round(temp.OD.max() + .5,1))

    plt.subplot(121)
    for ind,x in g2:
        x.sort_values('time',inplace=True)
        #plt.plot(x.time,patsy.build_design_matrices([y.design_info],x)[0],'k',alpha=.3)
        plt.plot(x.time,x.OD,'k',alpha=.1)
    plt.ylabel("log(OD)",fontsize=30)
    plt.xlabel("time (h)",fontsize=30)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    plt.grid(True,color='grey')
    plt.xlim(-2,44)
    plt.ylim(ylim)

    temp = g.get_group(strain)
    g2 = temp.groupby(['Experiment','Well'])

    plt.subplot(122)
    for ind,x in g2:
        x.sort_values('time',inplace=True)
        #plt.plot(x.time,patsy.build_design_matrices([y.design_info],x)[0],'g',alpha=.3)
        plt.plot(x.time,x.OD,'g',alpha=.1)

    plt.xlabel("time (h)",fontsize=30)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    plt.grid(True,color='grey')
    plt.xlim(-2,44)
    plt.ylim(ylim)
