import os
import numpy as np
import argparse
from data.nuscenes_pred_split import get_nuscenes_pred_split
from data.ethucy_split import get_ethucy_split
from utils.utils import print_log, AverageMeter, isfile, print_log, AverageMeter, isfile, isfolder, find_unique_common_from_lists, load_list_from_folder, load_txt_file


""" Metrics """

def compute_ADE(pred_arr, gt_arr, _):
    ade = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist.mean(axis=-1)                       # samples
        ade += dist.min(axis=0)                         # (1, )
    ade /= len(pred_arr)
    return ade


def compute_FDE(pred_arr, gt_arr, _):
    fde = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist[..., -1]                            # samples 
        fde += dist.min(axis=0)                         # (1, )
    fde /= len(pred_arr)
    return fde

def compute_avg_vel(pred_arr, _, curr_pos_arr):
    avgvel = 0.0
    for pred, curr_pos in zip(pred_arr, curr_pos_arr):
        # print("pred shape", pred.shape) # [num_sample, pred_step, 2]
        curr_pos = np.tile(curr_pos, (20, 1)).reshape(20,1,2)
        # print("curr pos shape", curr_pos.shape)
        last_step = np.concatenate((curr_pos, pred[:, :-1, :]), axis=1)
        vel_seq = pred - last_step
        vel_avg = np.sqrt(vel_seq[:,:,0] ** 2 + vel_seq[:,:,1] ** 2).mean()
        avgvel += vel_avg
    avgvel /= len(pred_arr)

    return avgvel

def compute_gt_avg_vel(_, gt_arr, curr_pos_arr):
    gt_avgvel = 0.0
    for gt, curr_pos in zip(gt_arr, curr_pos_arr):
        # print("pred shape", pred.shape) # [num_sample, pred_step, 2]
        gt_last_step = np.concatenate((curr_pos.reshape(1,2), gt[ :-1, :]), axis=0)
        gt_vel_seq = gt - gt_last_step
        gt_vel_avg = np.sqrt(gt_vel_seq[:,0] ** 2 + gt_vel_seq[:,1] ** 2).mean()
        gt_avgvel += gt_vel_avg
    gt_avgvel /= len(gt_arr)

    return gt_avgvel

def align_gt(pred, gt):
    frame_from_data = pred[0, :, 0].astype('int64').tolist()
    frame_from_gt = gt[:, 0].astype('int64').tolist()
    common_frames, index_list1, index_list2 = find_unique_common_from_lists(frame_from_gt, frame_from_data)
    assert len(common_frames) == len(frame_from_data)
    gt_new = gt[index_list1, 2:]
    curr_pos = gt[index_list1[0]-1, 2:]
    pred_new = pred[:, index_list2, 2:]
    return pred_new, gt_new, curr_pos


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nuscenes_pred')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--data', default='test')
    parser.add_argument('--log_file', default=None)
    args = parser.parse_args()

    dataset = args.dataset.lower()
    results_dir = args.results_dir
    
    if dataset == 'nuscenes_pred':   # nuscenes
        data_root = f'datasets/nuscenes_pred'
        gt_dir = f'{data_root}/label/{args.data}'
        seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
        seq_eval = globals()[f'seq_{args.data}']
    else:                            # ETH/UCY
        gt_dir = f'datasets/eth_ucy/{args.dataset}'
        seq_train, seq_val, seq_test = get_ethucy_split(args.dataset)
        seq_eval = globals()[f'seq_{args.data}']

    if args.log_file is None:
        log_file = os.path.join(results_dir, 'log_eval.txt')
    else:
        log_file = args.log_file
    log_file = open(log_file, 'a+')
    print_log('loading results from %s' % results_dir, log_file)
    print_log('loading GT from %s' % gt_dir, log_file)

    stats_func = {
        'ADE': compute_ADE,
        'FDE': compute_FDE,
        'avgvel': compute_avg_vel,
        'gt_avgvel': compute_gt_avg_vel
    }

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    seq_list, num_seq = load_list_from_folder(gt_dir)
    print_log('\n\nnumber of sequences to evaluate is %d' % len(seq_eval), log_file)
    for seq_name in seq_eval:
        # load GT raw data
        gt_data, _ = load_txt_file(os.path.join(gt_dir, seq_name+'.txt'))
        gt_raw = []
        for line_data in gt_data:
            line_data = np.array([line_data.split(' ')])[:, [0, 1, 13, 15]][0].astype('float32')
            if line_data[1] == -1: continue
            gt_raw.append(line_data)
        gt_raw = np.stack(gt_raw)

        data_filelist, _ = load_list_from_folder(os.path.join(results_dir, seq_name))    
            
        for data_file in data_filelist:      # each example e.g., seq_0001 - frame_000009
            # for reconsutrction or deterministic
            if isfile(data_file):
                # print("using reconstruction")
                all_traj = np.loadtxt(data_file, delimiter=' ', dtype='float32')        # (frames x agents) x 4
                all_traj = np.expand_dims(all_traj, axis=0)                             # 1 x (frames x agents) x 4
            # for stochastic with multiple samples
            elif isfolder(data_file):
                # print("using stochastic samples")
                sample_list, _ = load_list_from_folder(data_file)
                sample_all = []
                for sample in sample_list:
                    sample = np.loadtxt(sample, delimiter=' ', dtype='float32')        # (frames x agents) x 4
                    sample_all.append(sample)
                all_traj = np.stack(sample_all, axis=0)                                # samples x (framex x agents) x 4
            else:
                assert False, 'error'

            # convert raw data to our format for evaluation
            id_list = np.unique(all_traj[:, :, 1])
            frame_list = np.unique(all_traj[:, :, 0])
            agent_traj = []
            gt_traj = []
            curr_poss = []
            for idx in id_list:
                # GT traj
                gt_idx = gt_raw[gt_raw[:, 1] == idx]                          # frames x 4
                # print("gt_idx shape", gt_idx.shape)
                # predicted traj
                ind = np.unique(np.where(all_traj[:, :, 1] == idx)[1].tolist())
                pred_idx = all_traj[:, ind, :]                                # sample x frames x 4
                # filter data
                # print("gt_idx before alignment", gt_idx.shape)
                pred_idx, gt_idx, curr_pos = align_gt(pred_idx, gt_idx)
                # print("gt_idx after alignment", gt_idx.shape)
                # append
                agent_traj.append(pred_idx)
                gt_traj.append(gt_idx)
                curr_poss.append(curr_pos)

            """compute stats"""
            for stats_name, meter in stats_meter.items():
                func = stats_func[stats_name]
                value = func(agent_traj, gt_traj, curr_poss)
                meter.update(value, n=len(agent_traj))

            stats_str = ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in stats_meter.items()])
            print_log(f'eval seq {seq_name:s}, forecast fr. {int(frame_list[0]):06d} to {int(frame_list[-1]):06d} {stats_str}', log_file)

    print_log('-' * 30 + ' STATS ' + '-' * 30, log_file)
    for name, meter in stats_meter.items():
        print_log(f'{meter.count} {name}: {meter.avg:.4f}', log_file)
    print_log('-' * 67, log_file)
    log_file.close()
