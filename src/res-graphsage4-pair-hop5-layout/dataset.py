#print('import dataset.py ok !')
import os
import numpy as np
from glob import glob
import pickle

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *


OP_FEATURE = 140
OP_SIZE    = 128  #size of opcode dict
OP_CONFIG_FEATURE = 18
RUNTIME_MUL = 1e6

## helper function #################################################
class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

def read_pickle_from_file(pickle_file):
	with open(pickle_file,'rb') as f:
		x = pickle.load(f)
	return x

def write_pickle_to_file(pickle_file, x):
	with open(pickle_file, 'wb') as f:
		pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

####################################################################

def add_hop(r, num_hop=5):

	num_node = len(r['node_feat'])
	hop = np.full((num_node,), fill_value=-1, dtype=np.int8)
	hop[r['node_config_ids']] = 0

	for i in range(0, num_hop):
		current = np.where(hop != -1)[0]

		n = np.isin(r['edge_index'][:, 1], current)
		n = np.where(n)[0]
		n = r['edge_index'][n, 0]
		hop[n] = np.where(hop[n] == -1, i + 1, hop[n])

	r['hop'] = hop
	return r

def reduce_to_subgraph(r):

	subgraph_id = np.where(r['hop'] != -1)[0]

	n0 = np.isin(r['edge_index'][:, 0], subgraph_id)
	n1 = np.isin(r['edge_index'][:, 1], subgraph_id)
	edge_index = r['edge_index'][n0 & n1]

	num_node = len(r['node_feat'])
	node_id_to_subgraph_id = np.full((num_node,), fill_value=-1, dtype=np.int32)
	node_id_to_subgraph_id[subgraph_id] = np.arange(len(subgraph_id))
	subgraph_id_to_node_id = subgraph_id

	node_id_to_subgraph_id = {k: v for k, v in enumerate(node_id_to_subgraph_id)}
	subgraph_id_to_node_id = {k: v for k, v in enumerate(subgraph_id_to_node_id)}

	edge_index = np.array([[node_id_to_subgraph_id[i], node_id_to_subgraph_id[j]] for i, j in edge_index])
	node_config_ids = np.array([node_id_to_subgraph_id[i] for i in r['node_config_ids']])

	subgraph = dotdict(
		node_id_to_subgraph_id=node_id_to_subgraph_id,
		subgraph_id_to_node_id=subgraph_id_to_node_id,
		edge_index=edge_index,
		node_feat=r['node_feat'][subgraph_id],
		node_opcode=r['node_opcode'][subgraph_id],

		node_config_feat=r['node_config_feat'],
		node_config_ids=node_config_ids,
		config_runtime=r['config_runtime'],
	)
	return subgraph


def load_npz(dir, num_hop=5, to_tensor=True):

	data = []
	glob_file = glob(f'{dir}/*.npz')#[:3]
	glob_file = sorted(glob_file)
	for i, file in enumerate(glob_file):
		print('\r', f'{i}/{len(glob_file)}', file, end='')
		npz = np.load(file)
		r = dict(npz)
		# ['edge_index', 'node_feat', 'node_opcode', 'node_config_feat', 'node_config_ids', 'node_splits', 'config_runtime']

		if num_hop>0:
			r = add_hop(r, num_hop=num_hop)
			r = reduce_to_subgraph(r)

		r['file'] = file.split('/')[-1]
		r['num_node'] = len(r['node_opcode'])
		if to_tensor:
			r['edge_index'] = torch.tensor(r['edge_index'].T).long()
			r['node_opcode'] = torch.tensor(r['node_opcode']).long()
			r['node_feat'] = torch.tensor(r['node_feat']).float()
			r['node_config_ids'] = torch.tensor(r['node_config_ids']).long()
			r['node_config_feat'] = torch.tensor(r['node_config_feat']).float()
			r['config_runtime'] = torch.tensor(r['config_runtime']).float()

		data.append(r)
	print('')
	data = sorted(data, key=lambda d: -d['num_node'])
	return data


def do_subsample_data(data, size=100, is_random=False, is_droplast=True):
	subsample = []
	for i, r in enumerate(data):
		node_config_feat = r['node_config_feat']
		num_config = len(node_config_feat)
		idx = np.arange(num_config)

		if is_random:
			np.random.shuffle(idx)

		if is_droplast:
			idx = idx[:int(size * (num_config // size))]

		split = np.array_split(idx, max(1, len(idx) // size))
		for s in split:
			subsample.append((i, s.tolist()))

	return subsample


class LayoutDataset(Dataset):
	def __init__(self, data, subsample_param):
		self.data = data
		self.subsample_param = subsample_param
		self.subsample = None
		self.reset_subsample()

	def reset_subsample(self,):
		print('\r reset_subsample() ...')
		self.subsample  = do_subsample_data(self.data, **self.subsample_param)

	def __len__(self):
		return len(self.subsample)

	def __str__(self):
		string = ''
		string += f'\tlen = {len(self)}\n'
		string += f'\tdata = {len(self.data)}\n'
		return string

	def __getitem__(self, index):
		data_idx, subsample_idx = self.subsample[index]
		d = self.data[data_idx]

		r = {}
		r['index'] = index
		r['file'] = d['file']
		r['edge_index'] = d['edge_index']
		r['node_feat'] = d['node_feat']
		r['node_opcode'] = d['node_opcode']
		r['node_config_ids'] = d['node_config_ids']
		r['node_config_feat'] = d['node_config_feat'][subsample_idx]
		r['config_runtime'] = d['config_runtime'][subsample_idx]
		return r


tensor_key = ['edge_index', 'node_feat', 'node_opcode', 'node_config_feat', 'node_config_ids', 'config_runtime']  # 'node_splits',
def null_collate(batch):
	return batch


########################################################################
'''
#['edge_index', 'node_feat', 'node_opcode', 'node_config_feat', 'node_config_ids', 'node_splits', 'config_runtime']
layout
  |-nlp
  |  |-default :  test/train/valid
  |  |-random  :  test/train/valid  
  |  
  |-xla
     |-default :  test/train/valid   
     |-random  :  test/train/valid  

#['node_feat', 'node_opcode', 'edge_index', 'config_feat', 'config_runtime', 'config_runtime_normalizers']
tile
  |-xla :  test/train/valid   
  
'''


def run_make_mean_std_pickle_file(
	npz_dir,
	pickle_file,
):

	os.makedirs('/'.join(pickle_file.split('/')[:-1]), exist_ok=True)
	data = load_npz(npz_dir, num_hop=0, to_tensor=False)

	num_config, num_node = 0, 0
	config_feat, config_feat2 = 0, 0
	node_feat, node_feat2 = 0, 0

	for i in range(len(data)):
		r = data[i]
		# r = dict(r['npz'])

		c = r['node_config_feat'].reshape(-1, OP_CONFIG_FEATURE)
		n = r['node_feat'].reshape(-1, OP_FEATURE)

		num_config += len(c)
		num_node += len(n)
		config_feat += c.sum(0)
		config_feat2 += (c * c).sum(0)
		node_feat += n.sum(0)
		node_feat2 += (n * n).sum(0)

		#print(n.shape)
		#print(c.shape)
		#print('')

	node_feat_mean = node_feat / num_node
	node_feat2_mean = node_feat2 / num_node
	node_feat_std = np.sqrt(node_feat2_mean - node_feat_mean * node_feat_mean)

	config_feat_mean = config_feat / num_config
	config_feat2_mean = config_feat2 / num_config
	config_feat_std = np.sqrt(config_feat2_mean - config_feat_mean * config_feat_mean)

	# print(node_feat_mean.shape)
	# print(node_feat_std.shape)
	# print(config_feat_mean.shape)
	# print(config_feat_std.shape)
	# print('node_feat_mean\n', node_feat_mean)
	# print('node_feat_std\n', node_feat_std)
	# print('config_feat_mean\n', config_feat_mean)
	# print('config_feat_std\n', config_feat_std)
	write_pickle_to_file(
		pickle_file,
		{
			'node_feat_mean': node_feat_mean,
			'node_feat_std': node_feat_std,
			'node_config_feat_mean': config_feat_mean,
			'node_config_feat_std': config_feat_std,
		}
	)
	print('run_make_mean_std_pickle_file() ok!')


'''

'''


def run_check_dataset():
	root_dir = \
		'/home/titanx/hengck/share1/kaggle/2023/predict-ai-model-runtime'
	dir = \
		f'{root_dir}/data/predict-ai-model-runtime/npz_all/npz/layout/xla/default/train'

	data = load_npz(dir, num_hop=5, to_tensor=True)
	subsample_param = dict(size=10, is_random=True)
	dataset = LayoutDataset(data, subsample_param=subsample_param)
	print(dataset)


	for i in range(100):
		i = np.random.choice(len(dataset))
		r = dataset[i]
		print(f'index = {r["index"]}', '-----------')
		print(dataset.subsample[i])
		print('')

		# for k in ['config_feat','node_feat','node_opcode','edge_index','target']:
		for k in tensor_key:
			v = r[k].data.cpu().numpy()
			print(k)
			print('\t shape  :', v.shape)
			print('\t dtype  :', v.dtype)
			print('\t min/max:', v.min(), v.max())
			print('\t value  :', v.reshape(-1)[:5], '...')
		print('')


# main #################################################################
if __name__ == '__main__':
	run_check_dataset()