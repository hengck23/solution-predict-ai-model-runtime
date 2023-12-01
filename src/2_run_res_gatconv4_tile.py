import sys
sys.path.append('[third_party]')
sys.path.append('res-gatconv4-listmle-hop5-tile')

from dataset import *
from trainer import *
from submitter import *
from res_gatconv4_tile_configuration import *

##########################################################################################################################
for cfg in [
	tile_xla_cfg,
]:
	print('')
	print(f'# name = {cfg.name}')
	print('')

	print(f'# 1. make mean and std values for input feature ...')
	if 0:
		run_make_mean_std_pickle_file(
			npz_dir=f'{cfg.data_dir}/train',
			pickle_file=f'{cfg.pickle_file}'
		)

	print(f'# 2. train model  ...')
	if 1:
		run_train(cfg)

	print(f'# 3. local CV and submit  ...')
	if 1:
		run_submit(cfg, iteration='00010013')


