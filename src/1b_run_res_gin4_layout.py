import sys
sys.path.append('[third_party]')
sys.path.append('res-gin4-pair-hop5-layout')

from dataset import *
from trainer import *
from submitter import *
from res_gin4_layout_configuration import *

##########################################################################################################################
for cfg in [
	layout_xla_default_cfg,
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
		run_swa_and_submit(cfg)


