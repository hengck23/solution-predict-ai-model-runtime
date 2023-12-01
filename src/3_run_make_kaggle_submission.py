import pandas as pd
import numpy as np
import pickle
from scipy.stats import kendalltau
import os


## helper function #################################################
import res_gin4_layout_configuration
import sys
sys.modules['configure'] = res_gin4_layout_configuration

# class dotdict(dict):
# 	__setattr__ = dict.__setitem__
# 	__delattr__ = dict.__delitem__
# 	def __getattr__(self, name):
# 		try:
# 			return self[name]
# 		except KeyError:
# 			raise AttributeError(name)

def read_pickle_from_file(pickle_file):
	with open(pickle_file,'rb') as f:
		x = pickle.load(f)
	return x


####################################################################

def ensemble_pickle_to_csv(
	pickle_file,
	df_file,
	collection
):
	M = len(pickle_file)
	result = [
		read_pickle_from_file(pickle_file[m]) for m in range(M)
	]
	N = len(result[0])

	df_data = []
	for i in range(N):
		r = result[0][i]

		p = sum([result[m][i]['predict']for m in range(M)])
		argsort = np.argsort(p)
		top = ''.join([str(a) + ';' for a in argsort])
		top = top[:-1]

		file = r['file']
		file = file[:-4]
		d = {
			'ID': f'{collection}:{file}',
			'TopConfigs': top,
		}
		#print(d)
		df_data.append(d)

	df = pd.DataFrame(data=df_data)
	df.to_csv(df_file, index=False)
	print(df)
	print('')
	print(df.iloc[0])
	print('')

	if 1: # print ensemble score
		r1 = result[0]
		r2 = result[1]

		kendall_tau = []
		N = len(r1)
		for i in range(N):
			truth = r1[i].truth
			predict = r1[i].predict + r2[i].predict
			corr, _ = kendalltau(predict, truth)
			kendall_tau.append(corr)
			print(i, f'{corr:0.6f}', f'{r1[i]["file"]:128s}', )

		print('')
		kendall_tau = np.array(kendall_tau)
		opa_acc = (kendall_tau + 1) / 2
		print('kendall_tau', np.mean(kendall_tau))
		print('opa_acc', np.mean(opa_acc))
		print('')

	return df

###################################################
OUT_DIR = \
	'../results/final-01'
os.makedirs(f'{OUT_DIR}/ensemble', exist_ok=True)

for mode in ['valid','test']:
	ensemble_pickle_to_csv(
		pickle_file=[
			f'{OUT_DIR}/model/4x-graphsage-pair2/layout/xla-default/submit/layout-xla-default.swa-{mode}.pickle',
			f'{OUT_DIR}/model/4x-gin-pair2/layout/xla-default/submit/layout-xla-default.swa-{mode}.pickle'

		],
		df_file=f'{OUT_DIR}/ensemble/layout-xla-default.{mode}-ensemble.csv',
		collection='layout:xla:default',
	)

#-----
submit_file =f'{OUT_DIR}/submission_06.csv'
csv_file = [
	f'{OUT_DIR}/model/4x-graphsage-pair2/layout/nlp-default/submit/layout-nlp-default.swa-test.submit.csv',
	f'{OUT_DIR}/model/4x-graphsage-pair2/layout/nlp-random/submit/layout-nlp-random.swa-test.submit.csv',
	#f'{OUT_DIR}/model/4x-graphsage-pair2/layout/xla-default/submit/xla-nlp-default.swa-test.submit.csv',
	f'{OUT_DIR}/ensemble/layout-xla-default.{mode}-ensemble.csv',
	f'{OUT_DIR}/model/4x-graphsage-pair2/layout/xla-random/submit/layout-xla-random.swa-test.submit.csv',
	f'{OUT_DIR}/model/4x-gatconv-listmle/tile/xla/submit/tile-xla.00010013-test.submit.csv',
]

submit_df = pd.concat([pd.read_csv(f) for f in csv_file])
submit_df.loc[:,'ID']=submit_df['ID'].str.replace('/',':') #fix bug
submit_df.to_csv(submit_file, index=False)
print(submit_df)

