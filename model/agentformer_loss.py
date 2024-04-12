from eval import compute_ADE, compute_FDE, align_gt
import torch
from torch import nn


""" Oracles """

def oracle_prefers_smaller_de(z, pred, gt, compute_DE):
    des = []
    # pred: [z_dim, bs, fut_len, 2]; gt: [bs, fut_len, 2]
    bs = pred[0].shape[0] # get batch size
    for j in range(bs):
        de_one_samp = []
        for i in range(len(z)):
            prediction = pred[i][j]  # [fut_len, 2]
            ground_truth = gt[j,:,:] # [fut_len, 2]
            this_de = compute_DE(prediction.cpu().detach().numpy(),
                                    ground_truth.cpu().detach().numpy())
            de_one_samp.append(this_de)
        des.append(de_one_samp)
    des_tensor = torch.tensor(des) # [bs, z_dim]
    prefs = torch.zeros(bs)

    for i in range(bs):
        if des_tensor[i,0] < des_tensor[i,1]: # smaller is 'better'
            prefs[i] = 0.01
        elif des_tensor[i,0] > des_tensor[i,1]:
            prefs[i] = 0.99
        else:
            prefs[i] = 0.5
    return prefs

def oracle_prefers_smaller_de_batch(z, pred, gt, compute_DE):
    ''' For this, the multiple predictions are not used. 
    Instead, we sample random trajectories from the batch to compare '''
    pass

def oracle_prefers_slower_avg_vel(z, pref, gt):
    pass

def oracle_prefers_larger_mingap(z, pref, gt):
    pass    

""" Loss functions """
def compute_motion_mse(data, cfg):
    diff = data['fut_motion_orig'] - data['train_dec_motion']
    if cfg.get('mask', True):
        mask = data['fut_mask']
        diff *= mask.unsqueeze(2)
    loss_unweighted = diff.pow(2).sum() 
    if cfg.get('normalize', True):
        loss_unweighted /= diff.shape[0]
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist'].kl(data['p_z_dist']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_sample_loss(data, cfg):
    diff = data['infer_dec_motion'] - data['fut_motion_orig'].unsqueeze(1)
    if cfg.get('mask', True):
        mask = data['fut_mask'].unsqueeze(1).unsqueeze(-1)
        diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss_unweighted = dist.min(dim=1)[0]
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_oracle_preference_loss(data, cfg):

    z = data['oracle_eval_z']                                               # z   : [z_dim]; [bs, nz]
    pred = data['oracle_eval_dec_motion']                                   # pred: [z_dim]; [bs, fut_len, 2]; 
    gt = data['fut_motion_orig']                                            # gt  : [bs, fut_len, 2]
    
    oracles = cfg['oracles']
    used_oracles = 0
    prefs_list = []
    if oracles['ade']:
        pref_ade = oracle_prefers_smaller_de(z, pred, gt, compute_ADE)          # [bs]
        prefs_list.append(pref_ade)
        used_oracles += 1
    if oracles['fde']:
        pref_fde = oracle_prefers_smaller_de(z, pred, gt, compute_FDE)          # [bs]
        prefs_list.append(pref_fde)
        used_oracles += 1
    if oracles['ade_batch']:
        pref_ade_batch = oracle_prefers_smaller_de_batch(z, pred, gt, compute_ADE)
        prefs_list.append(pref_ade_batch)
        used_oracles += 1
    if oracles['fde_batch']:
        pref_fde_batch = oracle_prefers_smaller_de_batch(z, pred, gt, compute_FDE)
        prefs_list.append(pref_fde_batch)
        used_oracles += 1
    if oracles['avg_vel']:
        pref_avg_vel = oracle_prefers_slower_avg_vel(z, pred, gt)
        prefs_list.append(pref_avg_vel)
        used_oracles += 1
    if oracles['min_gap']:
        pref_mingap = oracle_prefers_larger_mingap(z, pred, gt)
        prefs_list.append(pref_mingap)
        used_oracles += 1

    if cfg['assign_tailored_dims']:
        #TODO: This is going to be delicate.
        pass
        loss_unweighted = 0
    elif cfg['assign_user_dims']:
        z_tensor = torch.stack((z[0], z[1]))                                    # [z_dim, bs, nz]
        # pref_all = torch.stack([pref_ade, pref_fde], dim=0).to(z_tensor.device) # [2, bs]
        pref_all = torch.stack(prefs_list, dim=0).to(z_tensor.device) # [used_oracles, bs]
        log_z_sm = torch.log(nn.functional.softmax(z_tensor, dim=0))

        assigned_dims = [0,24] # hardcoded for now

        loss_unweighted = 0
        N_times = 1 # For how many dimensions does each oracle control.
        for i in range(used_oracles):
            for j in range(N_times):
                this_loss = torch.sum(log_z_sm[1, :, assigned_dims[i] * N_times + j] * pref_all[i,:]) + \
                            torch.sum(log_z_sm[0, :, assigned_dims[i] * N_times + j] * (1-pref_all[i,:]))
                loss_unweighted += this_loss
        loss_unweighted = -loss_unweighted / (used_oracles * N_times)
    else:
        z_tensor = torch.stack((z[0], z[1]))                                    # [z_dim, bs, nz]
        # pref_all = torch.stack([pref_ade, pref_fde], dim=0).to(z_tensor.device) # [2, bs]
        pref_all = torch.stack(prefs_list, dim=0).to(z_tensor.device) # [used_oracles, bs]
        log_z_sm = torch.log(nn.functional.softmax(z_tensor, dim=0))

        # In this case, we use two metrics, and z_dim is essentially two as well (for now).
        # Hence, we use dims 0-9 for ADE and 10-19 for FDE for better usage of prefernce.
        # log_z_ade = log_z_sm[:,:,0:10]
        # log_z_fde = log_z_sm[:,:,10:20]

        # bp_loss_ade, bp_loss_fde = 0, 0

        # N_times = 10
        # for i in range(N_times):
        #     bp_loss_ade += (torch.sum(log_z_ade[1,:,i] * pref_all[0,:])
        #                     + torch.sum(log_z_ade[0,:,i] * (1-pref_all[0,:])))
        #     bp_loss_fde += (torch.sum(log_z_fde[1,:,i] * pref_all[1,:])
        #                     + torch.sum(log_z_fde[0,:,i] * (1-pref_all[1,:])))
        # loss_unweighted = -(bp_loss_ade + bp_loss_fde) / N_times
        
        loss_unweighted = 0
        N_times = 1 # For how many dimensions does each oracle control.
        for i in range(used_oracles):
            for j in range(N_times):
                this_loss = torch.sum(log_z_sm[1, :, i * N_times + j] * pref_all[i,:]) + \
                            torch.sum(log_z_sm[0, :, i * N_times + j] * (1-pref_all[i,:]))
                loss_unweighted += this_loss
        loss_unweighted = -loss_unweighted / (used_oracles * N_times)

    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


loss_func = {
    'mse': compute_motion_mse,
    'kld': compute_z_kld,
    'sample': compute_sample_loss,
    'op': compute_oracle_preference_loss
}