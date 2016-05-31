import pickle
import matplotlib.pyplot as plt
import pandas as pd

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
    
    

def analyze(g,ura3,k,name,eq,time_select,perm_ind,xslice=lambda x: x[:,1:],xslice_null=lambda x: x[:,2:],skip_strains=[],permutations=False,save=False,_pickle=False,plot=False):

    ind = ['strain','BF','BF-permuted']
    table = pd.DataFrame(columns=ind)

    for strain,temp in g:

        if strain in skip_strains:
            continue

        row = pd.Series(index=ind)
        row['strain'] = strain

        # add parent strain data
        temp_full = temp.append(ura3)

        select = temp_full.time.isin(time_select)
        temp = temp_full[select]

        y,x = patsy.dmatrices(eq,temp)

        print strain,temp.shape, temp.time.unique().shape

        gp = GPy.models.GPRegression(xslice(x),y,GPy.kern.RBF(k,ARD=True))
        gp.optimize()

        if plot:
            plt.figure(figsize=(12,6))
            plot_model(x,gp,strain)
            plot_data(temp_full,strain)

            ylim = (temp.OD.min(),temp.OD.max())

            plt.subplot(121)
            plt.title("$\Delta ura3$",fontsize=40)
            plt.ylim(ylim)
            plt.subplot(122)
            plt.title("$\Delta %s$"%strain,fontsize=40)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig("figures/%s/%s.png"%(name,strain),bbox_inches="tight",dpi=300)
            plt.close()

        gp_null = GPy.models.GPRegression(xslice_null(x),y,GPy.kern.RBF(k-1,ARD=True))
        gp_null.optimize()

        row.BF = gp.log_likelihood() - gp_null.log_likelihood()

        if _pickle:
            pickle.dump(gp,open('pickle/%s/%s.gp.pickle'%(name,strain),'w'))

        del gp

        if permutations:

            perms = []
            for i in range(50):
                if i%10 == 0:
                    print i

                x[:,perm_ind] = np.random.choice(x[:,perm_ind],x.shape[0],replace=False)
                gp = GPy.models.GPRegression(xslice(x),y,GPy.kern.RBF(k,ARD=True))
                gp.optimize()

                perms.append(gp.log_likelihood() - gp_null.log_likelihood())

                del gp

            row['BF-permuted'] = perms

        del gp_null

        if save:
            table = table.append(row,ignore_index=True)
            table.index = range(table.shape[0])
            table.to_csv("%s_bfs.csv"%name,index=False)

    return table
