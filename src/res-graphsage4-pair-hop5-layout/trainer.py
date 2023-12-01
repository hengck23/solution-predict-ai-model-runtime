import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from dataset import *
from model import *

import numpy as np
from timeit import default_timer as timer
import torch
from print_dict import format_dict

from my_lib.runner import *
from my_lib.file import *
from my_lib.net.rate import get_learning_rate

import sklearn.metrics
from scipy.stats import kendalltau

###############################################################################


def run_train(cfg):

	#--- setup ---
	seed_everything(cfg.seed)

	os.makedirs(cfg.fold_dir, exist_ok=True)
	for f in ['checkpoint', 'submit',]:
		os.makedirs(cfg.fold_dir + '/' + f, exist_ok=True)


	log = Logger()
	log.open(cfg.fold_dir + '/log.train.txt', mode='a')
	log.write(f'\n--- [START {log.timestamp()}] {"-" * 64}')
	log.write(f'__file__ = {__file__}\n')
	log.write(f'cfg:\n{format_dict(cfg)}')
	log.write(f'')


	#--- dataset ---
	train_data = load_npz(cfg.data_dir + '/train')
	valid_data = load_npz(cfg.data_dir + '/valid')

	train_dataset = LayoutDataset(train_data, subsample_param = dict(size=cfg.train_subsample_size, is_random=True, is_droplast=True))
	valid_dataset = LayoutDataset(valid_data, subsample_param = dict(size=cfg.valid_subsample_size, is_random=False, is_droplast=False))
	train_loader = DataLoader(
		train_dataset,
		sampler = RandomSampler(train_dataset),
		batch_size = cfg.train_batch_size,
		drop_last=True,
		num_workers=cfg.train_num_worker,
		#pin_memory=True,
		worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
		collate_fn=null_collate,
	)
	valid_loader = DataLoader(
		valid_dataset,
		sampler=SequentialSampler(valid_dataset),
		batch_size= cfg.valid_batch_size,
		drop_last=False,
		num_workers=cfg.valid_num_worker,
		#pin_memory=True,
		collate_fn=null_collate,
	)
	log.write(f'valid_dataset : \n{str(valid_dataset)}')
	log.write(f'train_dataset : \n{str(train_dataset)}')
	log.write('\n')


	#---model ---
	scaler = torch.cuda.amp.GradScaler(enabled=True)
	net = Net(cfg)
	net.cuda()

	optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
	log.write(f'optimizer:\n {str(optimizer)}')
	log.write('')


	#--- loop ---
	start_iteration = 0
	start_epoch = 0
	if cfg.resume_from.checkpoint is not None:
		f = torch.load(cfg.resume_from.checkpoint, map_location=lambda storage, loc: storage)
		state_dict = f['state_dict']
		print(net.load_state_dict(state_dict, strict=False))  # True
		if cfg.resume_from.iteration<0:
			start_iteration = f.get('iteration', 0)
			start_epoch = f.get('epoch', 0)

	num_train_batch = len(train_loader)
	iter_save  = int(cfg.epoch_save*num_train_batch)
	iter_valid = int(cfg.epoch_valid*num_train_batch)
	iter_log   = int(cfg.epoch_log*num_train_batch)
	train_loss = MyMeter([0, 0,], min(20,num_train_batch))#window must be less than num_train_batch
	valid_loss = [0, 0, 0]

	# logging
	def message_header():
		text = ''
		text += f'** start training here! **\n'
		text += f'                            |--------------- VALID--------|----- TRAIN/BATCH ---------------\n'
		text += f'rate      iter       epoch  |    -       opa      kendall |   loss   opa   |   time         \n'
		text += f'--------------------------------------------------------------------------------------------\n'
				 #1.00e-3  00000000*    0.00  |     0.00     0.00  |     0.00     0.00  |   0 hr 00 min
		text = text[:-1]
		return text

	def message(mode='print'):
		if mode == 'print':
			loss = batch_loss
		if mode == 'log':
			loss = train_loss

		if (iteration % iter_save == 0):
			asterisk = '*'
		else:
			asterisk = ' '

		lr =  get_learning_rate(optimizer)[0]
		lr =  short_e_format(f'{lr:0.2e}')

		timestamp =  time_to_str(timer() - start_timer, 'min')
		text = ''
		text += f'{lr}  {iteration:08d}{asterisk}  {epoch:6.2f}  |  '

		for v in valid_loss :
			text += f'{v:7.4f}  '
		text += f'|  '

		for v in loss :
			text += f'{v:5.2f}  '
		text += f'|  '

		#text += f'{loss[0]:7.2f}  {loss[1]:7.2f}  {loss[2]:4.3f}  {loss[3]:4.3f}  |  '
		text += f'{timestamp}'
		return text



	### start training here! ################################################

	def do_valid():
		result = {}
		num_valid = 0
		start_timer = timer()

		net.cuda()
		net.eval()
		net.output_type = ['loss', 'infer']
		for t, batch in enumerate(valid_loader):

			B = len(batch)
			assert(B==1)
			r = batch[0]

			for k in [
				'edge_index', 'node_feat', 'node_opcode', 'node_config_feat', 'node_config_ids', 'config_runtime']: #'node_splits',
				r[k] = r[k].cuda()

			with torch.cuda.amp.autocast(enabled=True):
				with torch.no_grad():
					config_runtime = net.infer_one(r)  # data_parallel(net,batch)

			num_valid += 1

			#-----
			index = r['index']
			data_idx, subsample_idx = valid_loader.dataset.subsample[index]

			if result.get(data_idx) is None:
				result[data_idx]=dotdict(
					file = r['file'],
					idx=[],
					truth=[],
					predict=[],
				)
			result[data_idx].idx.append(np.array(subsample_idx))
			result[data_idx].truth.append(r['config_runtime'].data.cpu().numpy())
			result[data_idx].predict.append(config_runtime.data.cpu().numpy())

			# -----
			print(f'\r validation: {num_valid}/{len(valid_dataset)}', time_to_str(timer() - start_timer, 'min'),
				  end='', flush=True)

		# ----
		#write_pickle_to_file('result.pickle',result)
		N = len(result)
		kendall_tau  = []
		for i in range(N):
			idx     = np.concatenate(result[i].idx)
			truth   = np.concatenate(result[i].truth)
			predict = np.concatenate(result[i].predict)
			corr, _ = kendalltau(predict, truth)
			kendall_tau.append(corr)

		kendall_tau = np.array(kendall_tau)
		opa_acc = (kendall_tau+1)/2

		valid_loss = [
			0,
			np.nanmean(opa_acc),
			np.nanmean(kendall_tau),
		]
		return valid_loss

	#---
	iteration   = start_iteration
	epoch       = start_epoch
	start_timer = timer()
	log.write(message_header())


	while epoch<cfg.num_epoch:
		train_loader.dataset.reset_subsample()
		for t, batch in enumerate(train_loader):

			# --- start of callback ---
			if iteration % iter_save == 0:
				if iteration != start_iteration:
					torch.save({
						'state_dict': net.state_dict(),
						'iteration': iteration,
						'epoch': epoch,
					}, f'{cfg.fold_dir}/checkpoint/{iteration:08d}.pth')
					pass

			#if 1:
			if iteration % iter_valid == 0:
				if iteration != start_iteration:
					valid_loss = do_valid()

			if (iteration % iter_log == 0) or (iteration % iter_valid == 0):
				print('\r', end='', flush=True)
				log.write(message(mode='log'))

			# --- end of callback ----

			net.train()
			net.output_type = ['loss', 'infer']

			optimizer.zero_grad()

			opa_acc_item = 0
			rank_loss_item = 0
			B = len(batch)
			for b in range(B):
				r = {}
				for k in ['edge_index', 'node_feat', 'node_opcode', 'node_config_feat', 'node_config_ids', 'config_runtime']: #'node_splits',
					r[k] = batch[b][k].cuda(non_blocking=True)

				with torch.cuda.amp.autocast(enabled=True):#True
					output = net([r])  # data_parallel(net,batch)
					opa_acc = output[f'opa_acc'].mean()/B
					rank_loss = output[f'rank_loss'].mean()/B

					opa_acc_item += opa_acc.item()
					rank_loss_item += rank_loss.item()

				scaler.scale(rank_loss).backward()

			# scaler.unscale_(optimizer)
			# torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
			scaler.step(optimizer)
			scaler.update()

			torch.clear_autocast_cache()
			batch_loss = [rank_loss_item,opa_acc_item]
			train_loss.step(batch_loss)

			print('\r', end='', flush=True)
			print(message(mode='print'), end='', flush=True)
			iteration +=  1
			epoch += 1/num_train_batch



