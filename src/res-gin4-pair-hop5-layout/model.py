from dataset import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
	SAGEConv,
	GINConv,
)

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

		hidden_dim = 256
		op_embed_dim = 128

		self.node_opcode_embed = nn.Embedding(OP_SIZE, op_embed_dim, max_norm=1)

		self.early_mlp = nn.Sequential(
			nn.Linear(op_embed_dim + OP_FEATURE + OP_CONFIG_FEATURE, hidden_dim, bias=False),
			MyNorm1d(hidden_dim),
			nn.GELU(),
			nn.Dropout(p=0.2),
			#nn.Linear(hidden_dim, hidden_dim, bias=False),
			#MyNorm1d(hidden_dim),
			#nn.GELU(),
		)
		#https://mlabonne.github.io/blog/posts/2022-04-25-Graph_Isomorphism_Network.html
		self.gconv = nn.ModuleList([
			GINConv(
				nn.Sequential(
					nn.Linear(hidden_dim, hidden_dim),
					MyNorm1d(hidden_dim),
					nn.ELU(),
					nn.Linear(hidden_dim, hidden_dim),
					MyNorm1d(hidden_dim),
					nn.ELU()
				)
			),
			GINConv(
				nn.Sequential(
					nn.Linear(hidden_dim, hidden_dim),
					MyNorm1d(hidden_dim),
					nn.ELU(),
					nn.Linear(hidden_dim, hidden_dim),
					MyNorm1d(hidden_dim),
					nn.ELU()
				)
			),
			GINConv(
				nn.Sequential(
					nn.Linear(hidden_dim, hidden_dim),
					MyNorm1d(hidden_dim),
					nn.ELU(),
					nn.Linear(hidden_dim, hidden_dim),
					MyNorm1d(hidden_dim),
					nn.ELU()
				)
			),
			GINConv(
				nn.Sequential(
					nn.Linear(hidden_dim, hidden_dim),
					MyNorm1d(hidden_dim),
					nn.ELU(),
					nn.Linear(hidden_dim, hidden_dim),
					MyNorm1d(hidden_dim),
					nn.ELU()
				)
			),
		])

		self.late_mlp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim, bias=False),
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
			output['rank_loss' ] = F_ordered_pair_loss(runtime, target)
			output['opa_acc'] = F_ordered_pair_accuracy(runtime, target)

		return output


	def infer_one(self, r):
		num_node   = len(r['node_feat'])  # 10
		num_config = len(r['node_config_feat'])  # 100

		node_opcode = self.node_opcode_embed(r['node_opcode'])  # 10, 32
		node_feat = r['node_feat']  # 10, 140
		node_feat = (node_feat - self.node_feat_mean) / (self.node_feat_std + 0.0001)


		# early fuse
		config_feat = r['node_config_feat']  # 100, 4, 18
		# config_feat = (config_feat - self.config_feat_mean) / (self.config_feat_std + 0.0001)
		config_to_node = F.one_hot(r['node_config_ids'], num_node).float()
		config_feat = torch.matmul(config_to_node.T.unsqueeze(0), config_feat)  # t 100, 10, 18

		x_early = torch.cat([
			node_opcode.unsqueeze(0).expand(num_config, -1, -1),
			node_feat.unsqueeze(0).expand(num_config, -1, -1),
			config_feat,
		], -1)

		edge_index = r['edge_index']
		x = self.early_mlp(x_early)  # 100, 10, 32


		x0 = self.gconv[0](x, edge_index)
		x1 = self.gconv[1](x0, edge_index)
		x2 = self.gconv[2](x1, edge_index)
		x3 = self.gconv[3](x1, edge_index)

		pool0 = torch.mean(x0,1)
		pool1 = torch.mean(x1,1)
		pool2 = torch.mean(x2,1)
		pool3 = torch.mean(x3,1)
		pool = pool0+pool1+pool2+pool3

		# late fuse ---
		x_late = self.late_mlp(pool)
		runtime = self.predict(x_late)
		runtime = runtime.squeeze(-1).float()
		return runtime


######################################################################################################3


def run_check_net():

	# dummy data for unit test
	def make_one():
		num_edge = 16
		num_node = 10
		num_config = 100
		num_config_id = 4

		r = dotdict(
			edge_index=torch.from_numpy(np.random.choice(num_node, (num_edge, 2)).T).long(),
			node_feat=torch.from_numpy(np.random.uniform(0, 1, (num_node, OP_FEATURE))).float(),
			node_opcode=torch.from_numpy(np.random.choice(OP_SIZE, (num_node,))).long(),
			node_config_ids=torch.from_numpy(np.random.choice(num_node, (num_config_id), replace=False)).long(),
			node_config_feat=torch.from_numpy(
				np.random.uniform(0, 1, (num_config, num_config_id, OP_CONFIG_FEATURE))).float(),
			config_runtime=torch.from_numpy(np.random.uniform(0, 1, (num_config,))).float() * RUNTIME_MUL,
		)
		return r

	batch = [make_one() for i in range(4)]

	# --------------------------------

	net = Net(cfg={
		'pickle_file':'/home/titanx/hengck/share1/kaggle/2023/predict-ai-model-runtime/result/final-01/npz-layout-nlp-default-train.stat.pickle',
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


