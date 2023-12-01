import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from timeit import default_timer as timer
import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.stats import kendalltau


from dataset import *
from model import *
from my_lib.file import *


def make_pickle_file(cfg, checkpoint_file, pickle_file, collection, mode):
    # --- dataset ---
    valid_data = load_npz(cfg.data_dir + f'/{mode}')
    valid_dataset = TileDataset(
        valid_data,
        subsample_param=dict(
            size=100,
            is_random=False,
            is_droplast=False
        )
    )

    # ---model ---
    net = Net(cfg)
    state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)['state_dict']
    print(net.load_state_dict(state_dict, strict=False))  # True

    net.cuda()
    net.eval()

    #-------------------------------------

    result = {}
    num_valid = len(valid_dataset)

    start_timer = timer()
    for index in range(num_valid):

        r = valid_dataset[index]
        for k in tensor_key:
            r[k] = r[k].cuda()

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                config_runtime = net.infer_one(r)  # data_parallel(net,batch)

        #-----
        data_idx, subsample_idx = valid_dataset.subsample[index]
        if result.get(data_idx) is None:
            result[data_idx] = dotdict(
                file=r['file'],
                idx=[],
                truth=[],
                predict=[],
            )
        result[data_idx].idx.append(np.array(subsample_idx))
        result[data_idx].truth.append(r['config_runtime'].data.cpu().numpy())
        result[data_idx].predict.append(config_runtime.data.cpu().numpy())
        # -----

        # show_result(batch, output, name='valid', wait=1)
        print(f'\r validation: {index}/{num_valid}', time_to_str(timer() - start_timer, 'min'),
              end='', flush=True)
    print('')
    #-----------------------------
    N = len(result)

    kendall_tau = []
    slowndown1_acc = []
    slowndown5_acc = []
    slowndown10_acc = []
    for i in range(N):
        result[i].idx = np.concatenate(result[i].idx)
        result[i].truth = np.concatenate(result[i].truth)
        result[i].predict = np.concatenate(result[i].predict)

        truth = result[i].truth
        predict = result[i].predict
        corr, _ = kendalltau(predict, truth)
        kendall_tau.append(corr)

        topk = np.argsort(predict)
        slowndown1_acc.append(2 - truth[topk[:1]].min() / truth.min())
        slowndown5_acc.append(2 - truth[topk[:5]].min() / truth.min())
        slowndown10_acc.append(2 - truth[topk[:10]].min() / truth.min())

        # print(i, result[i]['file'], corr)
        print(i, f'{slowndown5_acc[-1]:0.6f}', f'{corr:0.6f}', f'{result[i]["file"]:128s}', )

    print('')
    kendall_tau = np.array(kendall_tau)
    opa_acc = (kendall_tau + 1) / 2
    print('kendall_tau', np.mean(kendall_tau))
    print('opa_acc', np.mean(opa_acc))
    print('slowndown1_acc', np.mean(slowndown1_acc), 1 - np.mean(slowndown1_acc))
    print('slowndown5_acc', np.mean(slowndown5_acc), 1 - np.mean(slowndown5_acc))
    print('slowndown10_acc', np.mean(slowndown10_acc), 1 - np.mean(slowndown10_acc))
    print('')

    write_pickle_to_file(pickle_file, result)
    return result


def pickle_to_csv(pickle_file, df_file, collection):
    result = read_pickle_from_file(pickle_file)
    N = len(result)

    df_data = []
    for i in range(N):
        r = result[i]
        p = r['predict']
        argsort = np.argsort(p)[:5]
        top = ''.join([str(a) + ';' for a in argsort])
        top = top[:-1]

        file = r['file']
        file = file[:-4]  # .replace('.npz','')
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
    return df





##########################################################################
def run_submit(cfg, iteration):


    os.makedirs(f'{cfg.fold_dir}/submit', exist_ok=True)
    for mode in ['valid', 'test']:
        checkpoint_file = \
            f'{cfg.fold_dir}/checkpoint/{iteration}.pth'
        pickle_file = \
            f'{cfg.fold_dir}/submit/{cfg.collection.replace(":","-")}.{iteration}-{mode}.pickle'
        df_file =\
            f'{cfg.fold_dir}/submit/{cfg.collection.replace(":","-")}.{iteration}-{mode}.submit.csv'

        make_pickle_file(cfg, checkpoint_file, pickle_file, cfg.collection, mode)
        pickle_to_csv(pickle_file, df_file, cfg.collection)
