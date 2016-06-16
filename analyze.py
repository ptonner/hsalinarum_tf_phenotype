import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils, plot, patsy, GPy

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Analysis(object):

	def __init__(self,data,equation,p,label,derivative_ind,xslice=lambda x: x,xslice_null=lambda x: x,time=None,skip_strains=[],strains=None):
		self.data = data
		self.equation = equation
		self.p = p
		self.derivative_ind = derivative_ind
		self.label = label
		self.skip_strains = skip_strains
		self.xslice = xslice
		self.xslice_null = xslice_null
		if time is None:
			self.time = self.data.time.unique()
			self.time.sort()
		else:
			self.time = time
		self.current_strain = None
		
		if not strains is None:
			self._strains = strains
		else:
			self._strains = []

		self.full_time = np.linspace(self.time.min(),self.time.max())

		self.g = data.groupby('Strain')
		self.ura3 = self.g.get_group('ura3')
		self.g_ura3 = self.ura3.groupby(['Experiment','Well'])
		
	def use_strain(self,s):
		if len(self._strains) > 0:
			return s in self._strains
		return True

	def build_equation(self,):
		return 'OD ~ 0 + scale(time) + C(Strain,levels=l)' + self.equation

	def build_levels(self,strain):
		return ['ura3',strain]

	def build_strain_data(self,strain):
		temp = self.g.get_group(strain)
		temp_full = self.ura3.append(temp)

		select = temp_full.time.isin(self.time)
		temp = temp_full[select]

		l = self.build_levels(strain)
		eq = self.build_equation(); print eq
		y,x = patsy.dmatrices(eq,temp)

		return y,x

	def run(self,_plot=False,delta=False,_pickle=False,save=False,permutations=False,delta_kwargs={}):

		ind = ['strain','BF','BF-permuted']
		table = pd.DataFrame(columns=ind)
		self.od_delta = {}
		self.od_delta_deriv = {}

		for strain,temp in self.g:
			self.current_strain = strain

			if strain in self.skip_strains:
				continue
			elif not self.use_strain(strain):
				continue

			row = pd.Series(index=ind)
			row['strain'] = strain

			# add parent strain data
			temp_full = self.ura3.append(temp)

			select = temp_full.time.isin(self.time)
			temp = temp_full[select]

			l = self.build_levels(strain)
			y,x = patsy.dmatrices(self.build_equation(),temp)
			self.x = x

			# logger.info('%s: %s' % (strain, str(x.shape)))
			print '%s: %s' % (strain, str(x.shape))

			gp = GPy.models.GPRegression(self.xslice(x),y,self.build_kernel())
			gp.optimize()

			if _plot:
				self.plot_model(gp,x)

			if delta:
				self.delta(gp,x,strain,**delta_kwargs)

			del gp

		if self.delta:
			self.plot_deltas()

		self.current_strain = None

	def build_kernel(self):
		return GPy.kern.RBF(self.p,ARD=True)

	def xbase(self,strain=None,**kwargs):
		if strain is None:
			strain = self.current_strain
		ret = {'time':self.full_time,'Strain':[strain]*50}
		return ret

	def plot_model(self,gp,x):

		plt.figure(figsize=(12,6))
		plt.subplot(121)

		predx = patsy.build_design_matrices([x.design_info],self.xbase(strain='ura3'))[0]
		mu,var = gp.predict(predx[:,1:])
		mu = mu[:,0]
		var = var[:,0]

		plt.plot(self.full_time,mu,color='k')
		plt.fill_between(self.full_time,mu-2*np.sqrt(var),mu+2*np.sqrt(var),color='k',alpha=.2)

		plt.subplot(122)
		predx = patsy.build_design_matrices([x.design_info],self.xbase())[0]
		mu,var = gp.predict(predx[:,1:])
		mu = mu[:,0]
		var = var[:,0]

		plt.plot(self.full_time,mu,color='g')
		plt.fill_between(self.full_time,mu-2*np.sqrt(var),mu+2*np.sqrt(var),alpha=.2,color='g')

		plt.savefig("figures/%s/model/%s.png"%(self.label,self.current_strain),bbox_inches="tight")
		plt.close()

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

	def delta(self,gp,x,strain,x_1={},x_2={'Strain':['ura3']*50},xchange_2=lambda x: x,**kwargs):
		mu,var = utils.compute_delta(gp,x,
										self.xbase(),x_1,x_2,
										xslice=self.xslice,xchange_2=xchange_2)
		plt.figure(figsize=(12,6))
		plot.plot_mvn(mu,var,self.full_time)
		plt.plot([self.full_time[0],self.full_time[-1]],[0,0],'k',lw=3)
		plt.title("$\Delta$ log(OD)",fontsize=30)
		plt.xlabel("time (h)",fontsize=30)
		plt.yticks(fontsize=25)
		plt.xticks(fontsize=25)
		plt.grid(True,color='grey')
		plt.xlim(min(self.full_time)-2,max(self.full_time)+2)
		plt.savefig("figures/%s/od_delta/%s.png"%(self.label,strain),bbox_inches="tight")
		plt.close()
		self.od_delta[strain] = (mu,var)

		mu,var = utils.compute_delta(gp,x,
										 self.xbase(),x_1,x_2,derivative=True,derivative_ind=self.derivative_ind,
										 xslice=self.xslice,xchange_2=xchange_2)
		plt.figure(figsize=(12,6))
		plot.plot_mvn(mu,var,self.full_time)
		plt.plot([self.full_time[0],self.full_time[-1]],[0,0],'k',lw=3)
		plt.title("$\Delta$ d log(OD) / dt",fontsize=30)
		plt.xlabel("time (h)",fontsize=30)
		plt.yticks(fontsize=25)
		plt.xticks(fontsize=25)
		plt.grid(True,color='grey')
		plt.xlim(min(self.full_time)-2,max(self.full_time)+2)
		plt.savefig("figures/%s/od_delta_deriv/%s.png"%(self.label,strain),bbox_inches="tight")
		plt.close()
		self.od_delta_deriv[strain] = (mu,var)

	def plot_deltas(self):
		plt.figure(figsize=(36,.5*len(self.od_delta_deriv.keys())))
		plot.plot_delta(self.full_time,self.od_delta_deriv,mean=False,probability=True,cluster=True,plot_cluster=True,cluster_kwargs={"method":'complete'},ytick_filter=lambda x: "$\Delta %s$"%x)
		plt.yticks(fontsize=15)
		plt.savefig("figures/%s/od_delta_deriv_prob.png"%self.label,bbox_inches="tight",dpi=300)
		plt.close()


		plt.figure(figsize=(36,.5*len(self.od_delta_deriv.keys())))
		plot.plot_delta(self.full_time,self.od_delta_deriv,mean=True,probability=False,cluster=True,plot_cluster=True,cluster_kwargs={"method":'complete'},ytick_filter=lambda x: "$\Delta %s$"%x)
		plt.yticks(fontsize=15)
		plt.savefig("figures/%s/od_delta_deriv_mu.png"%self.label,bbox_inches="tight",dpi=300)
		plt.close()


		plt.figure(figsize=(36,.5*len(self.od_delta.keys())))
		plot.plot_delta(self.full_time,self.od_delta,mean=False,probability=True,cluster=True,plot_cluster=True,cluster_kwargs={"method":'complete'},ytick_filter=lambda x: "$\Delta %s$"%x)
		plt.yticks(fontsize=15)
		plt.savefig("figures/%s/od_delta_prob.png"%self.label,bbox_inches="tight",dpi=300)
		plt.close()


		plt.figure(figsize=(36,.5*len(self.od_delta.keys())))
		plot.plot_delta(self.full_time,self.od_delta,mean=True,probability=False,cluster=True,plot_cluster=True,cluster_kwargs={"method":'complete'},ytick_filter=lambda x: "$\Delta %s$"%x)
		plt.yticks(fontsize=15)
		plt.savefig("figures/%s/od_delta_mu.png"%self.label,bbox_inches="tight",dpi=300)
		plt.close()
