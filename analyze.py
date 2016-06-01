import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils, plot, patsy, GPy

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Analysis(object):

	def __init__(self,data,equation,p,label,derivative_ind,xslice=lambda x: x,xslice_null=lambda x: x,time=None,skip_strains=[]):
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
			self.time.sort(inplace=True)
		else:
			self.time = time
		self.current_strain = None

		self.full_time = np.linspace(self.time.min(),self.time.max())

		self.g = data.groupby('Strain')
		self.ura3 = self.g.get_group('ura3')
		self.g_ura3 = self.ura3.groupby(['Experiment','Well'])

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

			row = pd.Series(index=ind)
			row['strain'] = strain

			# add parent strain data
			temp_full = self.ura3.append(temp)

			select = temp_full.time.isin(self.time)
			temp = temp_full[select]

			l = self.build_levels(strain)
			y,x = patsy.dmatrices(self.build_equation(),temp)

			# logger.info('%s: %s' % (strain, str(x.shape)))
			print '%s: %s' % (strain, str(x.shape))

			gp = GPy.models.GPRegression(self.xslice(x),y,GPy.kern.RBF(self.p,ARD=True))
			gp.optimize()

			if delta:
				self.delta(gp,x,strain,**delta_kwargs)

			del gp

		self.current_strain = None

	def delta_xbase(self):
		return {'time':self.full_time}

	def delta(self,gp,x,strain,x_1={},x_2={},xchange_2=lambda x: x,**kwargs):
		mu,var = utils.compute_delta(gp,x,
										self.delta_xbase(),x_1,x_2,
										xslice=self.xslice,xchange_2=xchange_2)
		plt.figure(figsize=(12,6))
		plot.plot_mvn(mu,var)
		plt.title("$\Delta$ log(OD)",fontsize=30)
		plt.xlabel("time (h)",fontsize=30)
		plt.yticks(fontsize=25)
		plt.xticks(fontsize=25)
		plt.grid(True,color='grey')
		plt.xlim(-2,45)
		plt.savefig("figures/%s/od_delta/%s.png"%(self.label,strain),bbox_inches="tight")
		plt.close()
		self.od_delta[strain] = (mu,var)

		mu,var = utils.compute_delta(gp,x,
										 self.delta_xbase(),x_1,x_2,derivative=True,derivative_ind=self.derivative_ind,
										 xslice=self.xslice,xchange_2=xchange_2)
		plt.figure(figsize=(12,6))
		plot.plot_mvn(mu,var)
		plt.title("$\Delta$ d log(OD) / dt",fontsize=30)
		plt.xlabel("time (h)",fontsize=30)
		plt.yticks(fontsize=25)
		plt.xticks(fontsize=25)
		plt.grid(True,color='grey')
		plt.xlim(-2,45)
		plt.savefig("figures/%s/od_delta_deriv/%s.png"%(self.label,strain),bbox_inches="tight")
		plt.close()
		self.od_delta_deriv[strain] = (mu,var)

	def plot_deltas(self):
		plt.figure(figsize=(16,10))
		plot.plot_delta(self.full_time,self.od_delta_deriv,mean=False,probability=True,cluster=True,cluster_kwargs={"method":'complete'})
		plt.yticks(fontsize=15)
		plt.savefig("figures/%s/od_delta_deriv_prob.png"%self.label,bbox_inches="tight")
		plt.close()


		plt.figure(figsize=(16,10))
		plot.plot_delta(self.full_time,self.od_delta_deriv,mean=True,probability=True,cluster=True,cluster_kwargs={"method":'complete'})
		plt.yticks(fontsize=15)
		plt.savefig("figures/%s/od_delta_deriv_mu.png"%self.label,bbox_inches="tight")
		plt.close()


		plt.figure(figsize=(16,10))
		plot.plot_delta(self.full_time,self.od_delta,mean=False,probability=True,cluster=True,cluster_kwargs={"method":'complete'})
		plt.yticks(fontsize=15)
		plt.savefig("figures/%s/od_delta_prob.png"%self.label,bbox_inches="tight")
		plt.close()


		plt.figure(figsize=(16,10))
		plot.plot_delta(self.full_time,self.od_delta,mean=True,probability=True,cluster=True,cluster_kwargs={"method":'complete'})
		plt.yticks(fontsize=15)
		plt.savefig("figures/%s/od_delta_mu.png"%self.label,bbox_inches="tight")
		plt.close()
