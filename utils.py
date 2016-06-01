import pickle, patsy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def dict_copy(d1,d2):

	for k in d1.keys():
		d2[k] = d1[k]
	return d2

def gp_predict(gp,x,design_info):
	predx = patsy.build_design_matrices([x.design_info],{'time':time,'Strain':['ura3']*50,'paraquat':[0]*50})[0]
	mu,var = gp.predict(predx[:,1:])


def kernel_derivative(x,k,l):
	mult = [[((1./l)*(1-(1./l)*(y - z)**2))[0] for y in x] for z in x]
	return k*mult

def compute_delta(m,x,x_base={},x_1={},x_2={},function=True,derivative=False,derivative_ind=None,xslice=lambda x: x[:,1:],xchange_1=None,xchange_2=None):

	x_temp = dict_copy(x_base,{})
	x_temp = dict_copy(x_1,x_temp)

	predx = patsy.build_design_matrices([x.design_info],x_temp)[0]
	predx = xslice(predx)
	if derivative:
		mu_1,_ = m.predictive_gradients(predx)
		mu_1 = mu_1[:,derivative_ind,0]
		if function:
			_,var_1 = m._raw_predict(predx)
		else:
			temp,var_1 = m.predict(predx)
		var_1 = var_1[:,0];
		#var_1 = kernel_derivative(predx,var_1,m.kern.lengthscale[derivative_ind])[:,:,derivative_ind]; print var_1.shape
		var_1 = 1./m.kern.lengthscale[derivative_ind] * var_1
		if var_1.ndim > 1:
			var_1 = np.diag(var_1)
	else:
		if function:
			mu_1,var_1 = m._raw_predict(predx)
		else:
			mu_1,var_1 = m.predict(predx)
		mu_1 = mu_1[:,0]
		var_1 = var_1[:,0]

	x_temp = dict_copy(x_base,{})
	x_temp = dict_copy(x_2,x_temp)

	predx = patsy.build_design_matrices([x.design_info],x_temp)[0]
	predx = xslice(predx)
	if xchange_2:
		predx = xchange_2(predx)

	if derivative:
		mu_2,_ = m.predictive_gradients(predx)
		mu_2 = mu_2[:,derivative_ind,0]
		if function:
			_,var_2 = m._raw_predict(predx)
		else:
			_,var_2 = m.predict(predx)
		var_2 = var_2[:,0]
		var_2 = 1./m.kern.lengthscale[derivative_ind] * var_2
		if var_2.ndim > 1:
			var_2 = np.diag(var_2)
	else:
		if function:
			mu_2,var_2 = m._raw_predict(predx)
		else:
			mu_2,var_2 = m.predict(predx)
		mu_2 = mu_2[:,0]
		var_2 = var_2[:,0]

	return mu_1-mu_2, (np.sqrt(var_1) + np.sqrt(var_2))**2

# def analyze(g,ura3,k,name,eq,time_select,perm_ind,xslice=lambda x: x[:,1:],xslice_null=lambda x: x[:,2:],skip_strains=[],permutations=False,save=False,_pickle=False,plot=False):
#
#     ind = ['strain','BF','BF-permuted']
#     table = pd.DataFrame(columns=ind)
#
#     for strain,temp in g:
#
#         if strain in skip_strains:
#             continue
#
#         row = pd.Series(index=ind)
#         row['strain'] = strain
#
#         # add parent strain data
#         temp_full = temp.append(ura3)
#
#         select = temp_full.time.isin(time_select)
#         temp = temp_full[select]
#
#         y,x = patsy.dmatrices(eq,temp)
#
#         print strain,temp.shape, temp.time.unique().shape
#
#         gp = GPy.models.GPRegression(xslice(x),y,GPy.kern.RBF(k,ARD=True))
#         gp.optimize()
#
#         if plot:
#             plt.figure(figsize=(12,6))
#             plot_model(x,gp,strain)
#             plot_data(temp_full,strain)
#
#             ylim = (temp.OD.min(),temp.OD.max())
#
#             plt.subplot(121)
#             plt.title("$\Delta ura3$",fontsize=40)
#             plt.ylim(ylim)
#             plt.subplot(122)
#             plt.title("$\Delta %s$"%strain,fontsize=40)
#             plt.ylim(ylim)
#             plt.tight_layout()
#             plt.savefig("figures/%s/%s.png"%(name,strain),bbox_inches="tight",dpi=300)
#             plt.close()
#
#         gp_null = GPy.models.GPRegression(xslice_null(x),y,GPy.kern.RBF(k-1,ARD=True))
#         gp_null.optimize()
#
#         row.BF = gp.log_likelihood() - gp_null.log_likelihood()
#
#         if _pickle:
#             pickle.dump(gp,open('pickle/%s/%s.gp.pickle'%(name,strain),'w'))
#
#         del gp
#
#         if permutations:
#
#             perms = []
#             for i in range(50):
#                 if i%10 == 0:
#                     print i
#
#                 x[:,perm_ind] = np.random.choice(x[:,perm_ind],x.shape[0],replace=False)
#                 gp = GPy.models.GPRegression(xslice(x),y,GPy.kern.RBF(k,ARD=True))
#                 gp.optimize()
#
#                 perms.append(gp.log_likelihood() - gp_null.log_likelihood())
#
#                 del gp
#
#             row['BF-permuted'] = perms
#
#         del gp_null
#
#         if save:
#             table = table.append(row,ignore_index=True)
#             table.index = range(table.shape[0])
#             table.to_csv("%s_bfs.csv"%name,index=False)
#
#     return table
