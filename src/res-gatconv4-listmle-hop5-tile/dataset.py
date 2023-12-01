import os
import numpy as np
from glob import glob
import pickle

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import copy

OP_FEATURE = 140
OP_SIZE = 128  #size of opcode dict
OP_CONFIG_FEATURE = 18
CONFIG_FEATURE = 24
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


# https://discuss.pytorch.org/t/sorting-2d-tensor-by-pairs-not-columnwise/59465/6
def np_sort2d(a):
	inner_sorting = np.argsort(a[:, 0])  # here it does not matter whether we do it in stable way or not
	a_inner_sorted = a[inner_sorting]
	outer_sorting = np.argsort(a_inner_sorted[:, 1], kind='stable')
	a_outer_sorted = a_inner_sorted[outer_sorting]
	return a_outer_sorted


def load_npz(dir, to_tensor=True):

	data = []
	glob_file = glob(f'{dir}/*.npz')
	glob_file = sorted(glob_file)  # [:3]
	for i, file in enumerate(glob_file):
		print('\r', f'{i}/{len(glob_file)}', file, end='')
		npz = np.load(file)
		# layout: ['edge_index', 'node_feat', 'node_opcode', 'node_config_feat', 'node_config_ids', 'node_splits', 'config_runtime']
		# tile:   ['node_feat', 'node_opcode', 'edge_index', 'config_feat', 'config_runtime', 'config_runtime_normalizers']
		r = dict(npz)

		# ----
		r['file'] = file.split('/')[-1]
		r['num_node'] = len(r['node_opcode'])
		if to_tensor:
			r['edge_index'] = torch.tensor(np_sort2d(r['edge_index']).T).long()
			r['node_opcode'] = torch.tensor(r['node_opcode']).long()
			r['node_feat'] = torch.tensor(r['node_feat']).float()
			r['config_feat'] = torch.tensor(r['config_feat']).float()
			r['config_runtime'] = torch.tensor(r['config_runtime']).float()
			r['config_runtime_normalizers'] = torch.tensor(r['config_runtime_normalizers']).float()

		data.append(r)
	print('')
	data = sorted(data, key=lambda d: -d['num_node'])
	return data


def do_subsample_data(data, size=100, is_random=False, is_droplast=True):
	subsample = []
	for i, r in enumerate(data):
		node_config_feat = r['config_feat']
		num_config = len(node_config_feat)
		idx = np.arange(num_config)
		# todo: repeat if len less than size

		if is_random:
			np.random.shuffle(idx)

		if is_droplast:
			idx = idx[:int(size * max(1, num_config // size))]

		split = np.array_split(idx, max(1, len(idx) // size))
		for s in split:
			subsample.append((i, s.tolist()))

	return subsample


class TileDataset(Dataset):
	def __init__(self, data, subsample_param):
		self.data = data
		self.subsample_param = subsample_param
		self.subsample = None
		self.reset_subsample()

	def reset_subsample(self, ):
		print('\r reset_subsample() ...')
		self.subsample = do_subsample_data(self.data, **self.subsample_param)

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
		r['config_feat'] = d['config_feat'][subsample_idx]
		r['config_runtime_normalizers'] = d['config_runtime_normalizers'][subsample_idx]
		r['config_runtime'] = d['config_runtime'][subsample_idx]
		return r


tensor_key = ['node_feat', 'node_opcode', 'edge_index', 'config_feat', 'config_runtime', 'config_runtime_normalizers']
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
	data = load_npz(npz_dir, to_tensor=False)

	num_config, num_node = 0, 0
	node_feat, node_feat2 = 0, 0
	config_feat, config_feat2 = 0, 0

	for i in range(len(data)):
		r = data[i]
		# r = dict(r['npz'])
		c = r['config_feat'].reshape(-1, CONFIG_FEATURE)
		n = r['node_feat'].reshape(-1, OP_FEATURE)

		num_config += len(c)
		num_node += len(n)
		node_feat += n.sum(0)
		node_feat2 += (n * n).sum(0)

		config_feat += c.sum(0)
		config_feat2 += (c * c).sum(0)

		# print(n.shape)
		# print(c.shape)
		# print('')

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
			'config_feat_mean': config_feat_mean,
			'config_feat_std': config_feat_std,
		}
	)
	print('run_make_mean_std_pickle_file() ok!')




def run_check_dataset():
	root_dir = \
		'/home/titanx/hengck/share1/kaggle/2023/predict-ai-model-runtime'
	dir = \
		f'{root_dir}/data/predict-ai-model-runtime/npz_all/npz/tile/xla/train'

	data = load_npz(dir)
	dataset = TileDataset(data, subsample_param=dict(size=10, is_random=True))
	dataset.reset_subsample()
	print(dataset)

	for i in range(100):
		i = np.random.choice(len(dataset))
		r = dataset[i]

		print(f'index = {r["index"]}', '-----------')
		print(dataset.subsample[i])
		print('')
		for k in tensor_key:
			# for k in ['config_feat','node_feat','node_opcode','edge_index','target']:

			v = r[k].data.cpu().numpy()
			print(k)
			print('\t shape  :', v.shape)
			print('\t dtype  :', v.dtype)
			print('\t min/max:', v.min(), v.max())
			print('\t value  :', v.reshape(-1)[:5], '...')

		print('')


# main #################################################################
if __name__ == '__main__':
	run_check_dataset()  #