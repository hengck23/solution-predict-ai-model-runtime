from dataset import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
	SAGEConv,
	GATv2Conv,
)
from dataset import *
from myloss import *


'''
inspired by paper[1]

[1] GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training
https://github.com/lsj2408/GraphNorm

reference:
- https://github.com/lsj2408/GraphNorm/blob/master/GraphNorm_ws/gnn_ws/gnn_example/model/Norm/norm.py
- https://stackoverflow.com/questions/62670541/pytorch-instance-norm-implemented-by-basic-operations-has-different-result-comp
- https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/instancenorm.py

'''

class MyNorm1d(torch.nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.norm = nn.InstanceNorm1d(dim, affine=False)
		#Input: (N,C,L) or (C,L)
		self.bias= nn.Parameter(torch.zeros(dim))
		self.weight= nn.Parameter(torch.ones(dim))

	def compute_norm(self, x, axis):
		x_mean = torch.mean(x, axis=axis, keepdims=True)
		x_var = torch.var(x, axis=axis, keepdims=True, unbiased=False)
		x_norm = (x - x_mean) / torch.sqrt(x_var + 1e-5)
		return x_norm

	def forward(self, x):
		if x.ndim == 3:
			B, L, dim = x.shape
			norm = self.norm(x.permute(0,2,1)).permute(0,2,1)
			#norm=self.compute_norm(x, axis=(1,))
			#norm=norm*self.weight.reshape(1,1,-1) + self.bias.reshape(1,1,-1)
		if x.ndim == 2:
			B, dim = x.shape
			norm = self.norm(x.permute(1,0)).permute(1,0)
			#norm=self.compute_norm(x, axis=(0,))
			#norm=norm*self.weight.reshape(1,-1) + self.bias.reshape(1,-1)
		return norm


class Net(torch.nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.output_type = ['infer', 'loss']

		feat_stat = read_pickle_from_file(cfg['pickle_file'])

		self.node_feat_mean   = nn.Parameter(torch.from_numpy(feat_stat['node_feat_mean']).unsqueeze(0).float())
		self.node_feat_std    = nn.Parameter(torch.from_numpy(feat_stat['node_feat_std']).unsqueeze(0).float())
		self.config_feat_mean = nn.Parameter(torch.from_numpy(feat_stat['config_feat_mean']).unsqueeze(0).float())
		self.config_feat_std  = nn.Parameter(torch.from_numpy(feat_stat['config_feat_std']).unsqueeze(0).float())

		hidden_dim = 256
		op_embed_dim = 128

		self.node_opcode_embed = nn.Embedding(OP_SIZE, op_embed_dim, max_norm=1)

		self.early_mlp = nn.Sequential(
			nn.Linear(op_embed_dim + OP_FEATURE, hidden_dim, bias=False),
			MyNorm1d(hidden_dim),
			nn.GELU(),
			nn.Dropout(p=0.0),
			nn.Linear(hidden_dim, hidden_dim, bias=False),
			MyNorm1d(hidden_dim),
			nn.GELU(),
		)

		self.gconv = nn.ModuleList([
			GATv2Conv(hidden_dim, hidden_dim, ),
			GATv2Conv(hidden_dim, hidden_dim, ),
			GATv2Conv(hidden_dim, hidden_dim, ),
			GATv2Conv(hidden_dim, hidden_dim, ),
		])
		self.norm = nn.ModuleList([
			MyNorm1d(hidden_dim),
			MyNorm1d(hidden_dim),
			MyNorm1d(hidden_dim),
			MyNorm1d(hidden_dim),
		])

		self.late_mlp = nn.Sequential(
			nn.Linear(hidden_dim+CONFIG_FEATURE, hidden_dim, bias=False),
			MyNorm1d(hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, hidden_dim//2, bias=False),
			MyNorm1d(hidden_dim//2),
			nn.GELU(),
		)
		self.predict = nn.Linear(hidden_dim//2, 1)

	def forward(self, batch):
		runtime = []
		for r in batch:
			z = self.infer_one(r)
			runtime.append(z)
		runtime = torch.stack(runtime, 0)
		target  = torch.stack([r['config_runtime'].float()  for r in batch], 0)

		output = dotdict()
		if 'infer' in self.output_type:
			output['config_runtime'] = runtime

		if 'loss' in self.output_type:
			output['rank_loss' ] = F_listMLE(runtime, target)
			#output['rank_loss' ] = F_ordered_pair_loss(runtime, target)
			output['opa_acc'] = F_ordered_pair_accuracy(runtime, target)
			output['topk_acc'] = F_topk_slowdown(runtime, target)

		return output


	def infer_one(self, r):
		num_node   = len(r['node_feat'])  # 10
		num_config = len(r['config_feat'])  # 100

		node_opcode = self.node_opcode_embed(r['node_opcode'])  # 10, 32
		node_feat = r['node_feat']  # 10, 140
		node_feat = (node_feat - self.node_feat_mean) / (self.node_feat_std + 0.0001)

		x_early = torch.cat([
			node_opcode.unsqueeze(0),
			node_feat.unsqueeze(0),
		], -1)

		x = self.early_mlp(x_early)  # 100, 10, 32
		L,N,dim=x.shape
		x = x.reshape(L*N,dim)


		edge_index = r['edge_index']
		_2_,E = edge_index.shape
		edge_index = edge_index.unsqueeze(0).repeat(L,1,1)
		edge_index = edge_index + torch.arange(L).to(edge_index.device).reshape(L,1,1)*N
		edge_index = edge_index.permute(1,0,2)
		edge_index = edge_index.reshape(2,L*E)


		num_gconv = len(self.gconv)
		for i in range(num_gconv):
			x = self.gconv[i](x, edge_index) #+ x
			x = x.reshape(L, N, dim)
			x = self.norm[i](x)
			x = x.reshape(L * N, dim)
			x = F.gelu(x)
		x = x.reshape(L,N,dim)

		pool1   = torch.mean(x,1)
		pool2,_ = torch.max(x,1)
		pool    = pool1 + pool2

		#---
		config_feat = r['config_feat']
		config_feat = (config_feat - self.config_feat_mean) / (self.config_feat_std + 0.0001)
		x_late = torch.cat([
			config_feat, pool.repeat((num_config, 1))
		], axis=1)

		x_late = self.late_mlp(x_late)
		runtime = self.predict(x_late)
		runtime = runtime.squeeze(-1).float()

		return runtime


######################################################################################################3


# [ 'edge_index', 'node_feat', 'node_opcode', 'node_config_feat', 'node_config_ids',
#  'node_splits', 'config_runtime']

#https://discuss.pytorch.org/t/sorting-2d-tensor-by-pairs-not-columnwise/59465/6

def run_check_net():
	npz_dir = \
		'/home/titanx/hengck/share1/kaggle/2023/predict-ai-model-runtime/data/predict-ai-model-runtime/npz_all/npz/layout/xla/default/train/'
	npz_file = \
		f'{npz_dir}/alexnet_train_batch_32.npz'

	# npz = np.load(npz_file)

	# --------------------------------
	def make_one():
		num_edge = 16
		num_node = 10
		num_config = 100

		r = dotdict(
			edge_index=torch.from_numpy(np.random.choice(num_node, (num_edge, 2)).T).long(),
			node_feat=torch.from_numpy(np.random.uniform(0, 1, (num_node, OP_FEATURE))).float(),
			node_opcode=torch.from_numpy(np.random.choice(OP_SIZE, (num_node,))).long(),
			config_feat=torch.from_numpy(np.random.uniform(0, 1, (num_config, CONFIG_FEATURE))).float(),
			config_runtime=torch.from_numpy(np.random.uniform(0, 1, (num_config,))).float() * RUNTIME_MUL,
		)
		return r

	batch = [make_one() for i in range(4)]

	# --------------------------------

	net = Net(cfg={
		'pickle_file':'/home/titanx/hengck/share1/kaggle/2023/predict-ai-model-runtime/result/final-01a/npz-tile-xla-train.stat.pickle',
	})
	# print(net)

	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)
	# ---
	print('batch')
	r = batch[0]
	for k, v in r.items():
		print(f'{k:>32} : {v.shape} ')

	print('output')
	for k, v in output.items():
		if not any(l in k for l in ['loss', 'acc']):
			print(f'{k:>32} : {v.shape} ')
	print('loss')
	for k, v in output.items():
		if any(l in k for l in ['loss', 'acc']):
			print(f'{k:>32} : {v.item()} ')

# main #################################################################
if __name__ == '__main__':
	run_check_net()



