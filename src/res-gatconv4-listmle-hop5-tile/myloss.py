#https://github.com/allegro/allRank/blob/master/allrank/models/losses/listMLE.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_EPS = 1e-10

def F_listMLE(logit, truth, eps=DEFAULT_EPS):
	try:
		# shuffle for randomised tie resolution
		random_index = torch.randperm(logit.shape[-1])
		logit_shuffled = logit[:, random_index]
		truth_shuffled = truth[:, random_index]

		y_sorted, idx = truth_shuffled.sort(descending=True, dim=-1)
		l_sorted_by_y = torch.gather(logit_shuffled, dim=1, index=idx)

		l_max, _ = l_sorted_by_y.max(dim=1, keepdim=True)

		#https://effectivemachinelearning.com/PyTorch/7._Numerical_stability_in_PyTorch
		l_sorted_by_y = l_sorted_by_y - l_max #to prevent overflow
		cumsum = torch.cumsum(l_sorted_by_y.exp().flip(dims=[1]), dim=1).flip(dims=[1])
		loss = torch.log(cumsum + eps) - l_sorted_by_y

		loss = torch.mean(torch.sum(loss, dim=1))
	except:
		print('')
		print('error: F_listMLE()')
		print(logit)
		loss = logit*0
		loss = loss.sum()

	return loss

#https://www.tensorflow.org/ranking/api_docs/python/tfr/keras/metrics/OPAMetric

def F_ordered_pair_accuracy(score, truth):
	B, L = truth.shape

	score1 = score.reshape(B,1,L)
	score2 = score.reshape(B,L,1)
	s = score1>score2
	s = s.reshape(B,L*L)


	truth1 = truth.reshape(B,1,L)
	truth2 = truth.reshape(B,L,1)
	t = truth1>truth2
	t=t.reshape(B,L*L)

	index = torch.triu_indices(L, L, 1)
	mask = torch.ones((L,L),dtype=torch.bool).to(truth.device)
	mask = torch.triu(mask, diagonal=1)
	mask = mask.reshape(1,L*L).expand(B,-1)

	metric = (s*t*mask).sum()/(t*mask).sum()
	return metric

def F_ordered_pair_loss(score, truth):
	B, L = truth.shape

	score1 = score.reshape(B,1,L)
	score2 = score.reshape(B,L,1)
	s = score1-score2
	s = s.reshape(B,L*L)


	truth1 = truth.reshape(B,1,L)
	truth2 = truth.reshape(B,L,1)
	t = truth1>truth2
	t=t.reshape(B,L*L)

	index = torch.triu_indices(L, L, 1)
	mask = torch.ones((L,L),dtype=torch.bool).to(truth.device)
	mask = torch.triu(mask, diagonal=1)
	mask = mask.reshape(1,L*L).expand(B,-1)

	t = t[mask].float()
	s = s[mask].float()
	loss = -t*F.logsigmoid(s) -(1-t)*F.logsigmoid(-s)
	loss = loss.mean()


	return loss


#-----------------------------------------
def np_topk_slowdown(
	predict,
	runtime
):
	metric =[]
	num = len(runtime)
	for i in range(num):
		t = runtime[i]
		p = predict[i]

		top5 = np.argsort(p)[:5]
		m = 2- t[top5].min()/t.min()
		metric.append(m)

	return np.mean(metric)

def F_topk_slowdown(
	predict,
	runtime
):

	num = len(runtime)
	top5 = torch.argsort(predict,1)[:,:5]
	top5 = torch.stack([runtime[i][top5[i]] for i in range(num)])

	p, _ = top5.min(1)
	t, _ = runtime.min(1)
	slowdown = 2- p/t
	return torch.mean(slowdown)
