# DATA_DIR = \
# 	'/home/titanx/hengck/share1/kaggle/2023/predict-ai-model-runtime/data/predict-ai-model-runtime/npz_all/npz'
# OUT_DIR = \
# 	'/home/titanx/hengck/share1/kaggle/2023/predict-ai-model-runtime/result/final-01'

DATA_DIR = \
	'../data/predict-ai-model-runtime/npz_all/npz'
OUT_DIR = \
	'../results/final-01'


print(f'DATA_DIR={DATA_DIR}')
print(f'OUT_DIR={OUT_DIR}')


# https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
# pip install print-dict
from print_dict import format_dict
class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

# configuration
layout_nlp_default_cfg = dotdict(
	name = 'layout_nlp_default_cfg',

	# --- dataset ---
	collection   = 'layout:nlp:default',
	data_dir     = f'{DATA_DIR}/layout/nlp/default',
	pickle_file  = f'{OUT_DIR}/npz-layout-nlp-default-train.stat.pickle',
	data_num_hop = 5,
	train_subsample_size=80,
	valid_subsample_size=100,

	train_num_worker=4,
	train_batch_size=32,
	valid_num_worker=1,
	valid_batch_size=1,

	# --- model ---
	lr=0.0005,

	# --- loop ---
	resume_from=dotdict(
		iteration=-1,
		checkpoint=None,
	),
	num_epoch=2.5,
	epoch_save=0.1,
	epoch_valid=0.1,
	epoch_log=0.1,

	# --- experiment ---
	seed=123,
	fold_dir =
		f'{OUT_DIR}/model/4x-graphsage-pair2/layout/nlp-default',

	# --- swa ---
	swa_min='00004980',
	swa_max='00010458',
)

#-----

layout_nlp_random_cfg = dotdict(
	name = 'layout_nlp_random_cfg',

	# --- dataset ---
	collection   = 'layout:nlp:random',
	data_dir     = f'{DATA_DIR}/layout/nlp/random',
	pickle_file  = f'{OUT_DIR}/npz-layout-nlp-default-train.stat.pickle',
	data_num_hop = 5,
	train_subsample_size=80,
	valid_subsample_size=100,

	train_num_worker=4,
	train_batch_size=32,
	valid_num_worker=1,
	valid_batch_size=1,

	# --- model ---
	lr=0.0005,

	# --- loop ---
	resume_from=dotdict(
		iteration=-1,
		checkpoint=None,
	),
	num_epoch=1.2,
	epoch_save=0.1,
	epoch_valid=0.1,
	epoch_log=0.1,

	# --- experiment ---
	seed=123,
	fold_dir =
		f'{OUT_DIR}/model/4x-graphsage-pair2/layout/nlp-random',

	# --- swa ---
	swa_min='00003675',
	swa_max='00005775',
)


#=====

layout_xla_default_cfg = dotdict(
	name = 'layout_xla_default_cfg',

	# --- dataset ---
	collection   = 'layout:xla:default',
	data_dir     = f'{DATA_DIR}/layout/xla/default',
	pickle_file  = f'{OUT_DIR}/npz-layout-xla-default-train.stat.pickle',
	data_num_hop = 5,
	train_subsample_size=32, #48
	valid_subsample_size=100,

	train_num_worker=4,
	train_batch_size=32,
	valid_num_worker=1,
	valid_batch_size=1,

	# --- model ---
	lr=0.0005,

	# --- loop ---
	resume_from=dotdict(
		iteration=-1,
		checkpoint=None,
	),
	num_epoch=2.2,
	epoch_save=0.1,
	epoch_valid=0.1,
	epoch_log=0.1,

	# --- experiment ---
	seed=123,
	fold_dir =
		f'{OUT_DIR}/model/4x-graphsage-pair2/layout/xla-default',

	# --- swa ---
	swa_min='00003060',
	swa_max='00005100',
)



layout_xla_random_cfg = dotdict(
	name = 'layout_xla_random_cfg',

	# --- dataset ---
	collection   = 'layout:xla:random',
	data_dir     = f'{DATA_DIR}/layout/xla/random',
	pickle_file  = f'{OUT_DIR}/npz-layout-xla-default-train.stat.pickle',
	data_num_hop = 5,
	train_subsample_size=100, #48
	valid_subsample_size=200,

	train_num_worker=4,
	train_batch_size=32,
	valid_num_worker=1,
	valid_batch_size=1,

	# --- model ---
	lr=0.0005,

	# --- loop ---
	resume_from=dotdict(
		iteration=0,
		checkpoint=f'{OUT_DIR}/model/4x-graphsage-pair2/nlp-random-old/checkpoint/00000975.pth',
	),
	num_epoch=2.2,
	epoch_save=0.1,
	epoch_valid=0.1,
	epoch_log=0.1,

	# --- experiment ---
	seed=123,
	fold_dir =
		f'{OUT_DIR}/model/4x-graphsage-pair2/layout/xla-random',

	# --- swa ---
	swa_min='00000341',
	swa_max='00000558',
)
