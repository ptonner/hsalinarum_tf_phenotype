import pandas as pd
import numpy as np

def loadData(strains=[],standard=False,paraquat=False,osmotic=False,heatshock=False,peroxide=False,mean=False,scaleX=True,batchEffects=False,nanRemove=True,plates=None):
	import os
	datadir = 'data'
	# print datadir
	# print os.path.join(datadir,"hsalinarum/tidy_normalize_log_st0.csv")

	if batchEffects:
		data = pd.read_csv(os.path.join(datadir,"tidy_normalize_log_st0_batchEffects.csv"),index_col=None)
	else:
		data = pd.read_csv(os.path.join(datadir,"tidy_normalize_log_st0.csv"),index_col=None)

	if not plates is None:
		data = data.loc[data.Experiment.isin(plates)]

	conditions = ['Experiment','Well','Strain','standard','paraquat','osmotic','heatshock','peroxide']
	temp = data.set_index(conditions+['time'])
	temp = temp[['OD']]

	if len(strains)==0:
		strains = ['ura3', 'hlx1', 'asnC', 'trh2', 'trh3', 'trh4', 'copR', 'kaiC',
       'idr1', 'idr2', 'troR', 'phoU', 'prp2', 'birA', 'trmB', 'arcR',
       'VNG0039', 'VNG2268', 'VNG0471', 'VNG1029', 'VNG2614', 'rosR',
       'hlx2', 'cspD1', 'cspD2', 'sirR', 'VNG0194H', 'hrg']

	# put data in s x n shape, with s samples and n timepoints
	pivot = temp.unstack(-1)
	pivot.columns = [t for s,t in pivot.columns.values]

	effects = []
	otherEffects = []
	selectStrain = pivot.index.get_level_values('Strain').isin(strains)
	selectCondition = pd.Series([False]*pivot.shape[0],index=pivot.index)
	if standard:
		selectCondition = selectCondition | (pivot.index.get_level_values('standard')==1)
		effects+=['standard']
	if paraquat:
		selectCondition = selectCondition | (pivot.index.get_level_values('paraquat')==1)
		effects+=['paraquat']
	if osmotic:
		selectCondition = selectCondition | (pivot.index.get_level_values('osmotic')==1)
		effects+=['osmotic']
	if heatshock:
		selectCondition = selectCondition | (pivot.index.get_level_values('heatshock')==1)
		effects+=['heatshock']
	if peroxide:
		selectCondition = selectCondition | (pivot.index.get_level_values('peroxide')==1)
		effects+=['peroxide']
	select = selectStrain & selectCondition
	pivot = pivot.loc[select,:]

	if mean:
		new_pivot = []
		new_index = []
		for s in strains:
			for e in effects:
				ni = [s]+[0]*len(effects)
				ni[effects.index(e)+1] = 1

				select = (pivot.index.get_level_values('Strain')==s) & (pivot.index.get_level_values(e)==1)
				temp = pivot.loc[select,:]
				#print s,e,temp.shape[0]
				new_index.append(tuple(ni))
				new_pivot.append(temp.mean(0).values)
		new_pivot = np.array(new_pivot)
		pivot = pd.DataFrame(new_pivot,index=pd.MultiIndex.from_tuples(new_index,names=['Strain']+effects),columns=pivot.columns)

	fact,labels = pd.factorize(pivot.index.get_level_values('Strain'))
	e = []
	for eff in effects:
		e.append(pivot.index.get_level_values(eff))
	e = np.array(e).T
	e = np.where(e!=0)[1]

	if len(otherEffects)>0:
		e2 = []
		for eff in otherEffects:
			e2.append(pd.factorize(pivot.index.get_level_values(eff))[0])
		e2 = np.array(e2).T
		e = np.array([fact,e,e2]).T
	else:
		if len(effects) <= 1:
			e = np.array(fact)[:,None]
		else:
			e = np.array([fact,e,]).T

	x = pivot.columns.values

	if scaleX:
		x = (x-x.mean())/x.std()
	x = x[:,None]

	y = pivot.values.T

	if nanRemove:
		select = ~np.any(np.isnan(y),1)
		y = y[select,:]
		x = x[select,:]

	return x,y,e,labels


if __name__ == "__main__":
	import argparse, sys, os, gpfanova, logging

	logging.basicConfig(filename='runFANOVA.log',level=logging.DEBUG)

	parser = argparse.ArgumentParser(description='Run analysis of H. salinarum TF data.')
	parser.add_argument('strains',metavar=('s'), type=str, nargs='*',
	                  help='strains to build model for')
	parser.add_argument('-n', dest='n_samples', action='store',default=10, type=int,
	                   help='number of samples to generate from posterior')
	parser.add_argument('-t', dest='thin', action='store',default=10, type=int,
	                   help='thinning rate for the posterior')
	parser.add_argument('--label', dest='label', action='store',default='', type=str,
	                   help='add a label to this run')
	parser.add_argument('--plates', dest='plates', action='store',default='', type=str,
	                   help='plates to use in this run')
	parser.add_argument('-i', dest='interactions', action='store_true',
	                   help='include interactions in the model')
	parser.add_argument('-g', dest='generateCommands', action='store_true',
	                   help='generate the commands for this script')
	# parser.add_argument('-a', dest='analyze', action='store_true',
	#                    help='analyze the output of this script')
	parser.add_argument('-s', dest='standard', action='store_true',
	                   help='analyze standard data')
	parser.add_argument('-p', dest='paraquat', action='store_true',
	                   help='analyze paraquat data')
	parser.add_argument('-o', dest='osmotic', action='store_true',
	                   help='analyze osmotic data')
	parser.add_argument('-e', dest='heatshock', action='store_true',
	                   help='analyze heatshock data')
	parser.add_argument('-x', dest='peroxide', action='store_true',
	                   help='analyze peroxide data')
	parser.add_argument('--helmertConvert', dest='helmertConvert', action='store_true',
	                   help='helmertConvert toggle for gpfanova')
	parser.add_argument('--scaleX', dest='scaleX', action='store_true',
	                   help='scaleX toggle for data')
	parser.add_argument('--batchEffects', dest='batchEffects', action='store_true',
	                   help='batchEffects toggle for data')
	parser.add_argument('--coprCompliment', dest='coprCompliment', action='store_true',
	                   help='run coprCompliment analysis')
	parser.add_argument('-m', dest='mean', action='store_true',
	                   help='convert data to mean')

	args = parser.parse_args()

	if args.generateCommands:
		print '\n'.join(generate_commands(args.n_samples,args.interactions))
	# elif args.analyze:
	# 	analyze()
	else:

		if not args.plates == '':
			plates = args.plates.split(",")

		if args.coprCompliment:

			x,y,effect,labels = loadData(['ura3','ura3+pMTFcmyc','VNG1179C+pMTFcmyc','VNG1179C-VNG1179C','copR'],
							standard=args.standard,paraquat=args.paraquat,osmotic=args.osmotic,heatshock=True,
							mean=args.mean,scaleX=args.scaleX,batchEffects=args.batchEffects,nanRemove=True,
							plates=['heatshock_12'])

			import numpy as np

			# columns: strain, ev, copr-vector
			neweffects = np.zeros((5,3),dtype=int)
			neweffects[labels.str.contains("VNG1179"),0] = 1
			neweffects[labels.str.contains("copR"),0] = 1
			neweffects[labels.str.contains("\+"),1] = 1
			neweffects[labels=='VNG1179C-VNG1179C',[1,2]] = 1
			effect = neweffects[effect[:,0],:]

		else:
			x,y,effect,_ = loadData(args.strains,standard=args.standard,paraquat=args.paraquat,osmotic=args.osmotic,heatshock=args.heatshock,peroxide=args.peroxide,
								mean=args.mean,scaleX=args.scaleX,batchEffects=args.batchEffects,nanRemove=True)

		m = gpfanova.fanova.FANOVA(x,y,effect,interactions=args.interactions,helmertConvert=args.helmertConvert)

		# resultsDir = os.path.abspath(os.path.join(os.path.abspath(__file__),'results'))
		resultsDir = 'results'

		s = 'posteriorSamples'
		if args.coprCompliment:
			s += "_coprCompliment"
		if args.interactions:
			s += '_interactions'
		if args.helmertConvert:
			s += "_helmertConvert"
		if args.mean:
			s+= '_mean'
		if args.scaleX:
			s+= '_scaleX'
		if args.batchEffects:
			s+= '_batchEffects'
		s+= "_"
		temp = ''
		if args.standard:
			s += temp + "standard"
			temp = '-'
		if args.paraquat:
			s += temp + "paraquat"
			temp = '-'
		if args.osmotic:
			s += temp + "osmotic"
			temp = '-'
		if args.heatshock:
			s += temp + "heatshock"
			temp = '-'
		if args.peroxide:
			s += temp + "peroxide"
			temp = '-'

		if len(args.strains)>0:
			s += '_(%s)'%",".join(args.strains)

		if args.label!="":
			s += "_%s"%args.label

		nrestarts = 0
		while nrestarts < 10:
			try:
				m.sample(args.n_samples,args.thin,verbose=True)
				break
			except Exception,e:
				m.save(os.path.join(resultsDir,'%s.csv'%(s)))
				nrestarts+=1

				# try walking back the sampler
				m.parameter_cache = m.parameter_history.iloc[-1,:]

				print nrestarts, e

		m.save(os.path.join(resultsDir,'%s.csv'%(s)))
