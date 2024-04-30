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
    de_mask = torch.ones(bs)

    return prefs, de_mask

def oracle_prefers_smaller_de_batch(z, pred, gt, compute_DE):
    ''' For this, the multiple predictions are not used. 
    Instead, we sample random trajectories from the batch to compare '''
    pass

def oracle_prefers_slower_avg_vel(z, pred, gt, pre_motion, mask=False):
    des1, des2 = None, None
    # pred: [z_dim, bs, fut_len, 2]; pre_motion: [fut_len, bs, 2]
    pre_motion = pre_motion.permute(1,0,2) # [bs, fut_len, 2]
    bs = pred[0].shape[0] # get batch size

    for i in range(len(z)):
        vel_seqs = pred[i] - torch.cat([pre_motion[:,-1,:].unsqueeze(1), pred[i][:, :-1, :]],dim=1)
        vel_avg = torch.sqrt(vel_seqs[:,:,0].pow(2) + vel_seqs[:,:,1].pow(2)).mean(dim=1) # [bs]

        # print("vel_avg", vel_avg.shape)
        if i == 0:
            des1 = vel_avg
        else:
            des2 = vel_avg
    prefs = torch.zeros(bs)
    for i in range(bs):
        z0 = z[0][i,0]
        z1 = z[1][i,0]
        if des1[i] < des2[i]: # smaller is 'better'
            # prefs[i] = 0.01
            prefs[i] = z0/(z0+z1)
        elif des1[i] > des2[i]:
            # prefs[i] = 0.99
            prefs[i] = z1/(z0+z1)
        else:
            prefs[i] = 0.5

    vel_mask = torch.ones(bs) # All ones. This means no mask.
    if mask:
        used = torch.ones(bs)
        # gt_vel = 0.3525 # ground truth for eth
        gt_vel_seq = gt - torch.cat([pre_motion[:,-1,:].unsqueeze(1), gt[:, :-1, :]],dim=1)
        gt_vel = torch.sqrt(gt_vel_seq[:,:,0].pow(2) + gt_vel_seq[:,:,1].pow(2)).mean(dim=1) # [bs]
        #TODO: change to grount truth of datapoint
        ## Comparative mask
        # Only consider those prediction pairs (y1e, y2e) that (y1e-y_gt) * (y2e-y_gt) < 0 
        # (one smaller than ground truth, one larger)
        for i in range(bs):
            if (des1[i] - gt_vel[i]) * (des2[i] - gt_vel[i]) >= 0:
                vel_mask[i] = 0
                used[i] = 0
        ########## END of Comparative mask ##########

        ## Percentage mask
        # Only consider those prediction pairs (y1e, y2e) that |(yie-y_gt) / y_gt| > 5%
        percentage_threshold = 0.1
        for i in range(bs):
            if abs(des1[i] - gt_vel[i]) / gt_vel[i] < percentage_threshold \
                or abs(des2[i] - gt_vel[i]) / gt_vel[i] < percentage_threshold:
                vel_mask[i] = 0
                used[i] = 0
        
        used_cnt = torch.sum(used)
        ########## END of Percentage mask ##########
    return prefs, vel_mask, used_cnt

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
    pre_motion = data['pre_motion']                                         # pre_motion: [fut_len, bs, 2]  

    oracles = cfg['oracles']
    used_oracles = 0
    prefs_list = []
    mask_list = []

    if oracles['ade']:
        pref_ade, ade_mask = oracle_prefers_smaller_de(z, pred, gt, compute_ADE)          # [bs]
        prefs_list.append(pref_ade)
        mask_list.append(ade_mask)
        used_oracles += 1
    if oracles['fde']:
        pref_fde, fde_mask = oracle_prefers_smaller_de(z, pred, gt, compute_FDE)          # [bs]
        prefs_list.append(pref_fde)
        mask_list.append(fde_mask)
        used_oracles += 1
    if oracles['ade_batch']:
        pref_ade_batch,_ = oracle_prefers_smaller_de_batch(z, pred, gt, compute_ADE)
        prefs_list.append(pref_ade_batch)
        used_oracles += 1
    if oracles['fde_batch']:
        pref_fde_batch,_ = oracle_prefers_smaller_de_batch(z, pred, gt, compute_FDE)
        prefs_list.append(pref_fde_batch)
        used_oracles += 1
    if oracles['avg_vel']:
        pref_avg_vel, vel_mask, used = oracle_prefers_slower_avg_vel(z, pred, gt, pre_motion, mask=cfg['mask_insignificant_comp'])
        prefs_list.append(pref_avg_vel)
        mask_list.append(vel_mask)
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
        z_tensor = torch.stack((z[0], z[1]))                                        # [z_dim, bs, nz]
        # pref_all = torch.stack([pref_ade, pref_fde], dim=0).to(z_tensor.device) # [2, bs]
        pref_all = torch.stack(prefs_list, dim=0).to(z_tensor.device)               # [used_oracles, bs]
        mask_all = torch.stack(mask_list, dim=0).to(z_tensor.device)                # [used_oracles, bs]
        # log_z_sm = torch.log(nn.functional.softmax(z_tensor, dim=0))                # [z_dim, bs, nz]
        z_tensor_norm = torch.stack((z_tensor[0]/(z_tensor[0]+z_tensor[1]), z_tensor[1]/(z_tensor[0]+z_tensor[1])))
        log_z_sm = torch.log(z_tensor_norm)

        assigned_dims = [0] # hardcoded for now

        loss_unweighted = 0
        N_times = 1 # For how many dimensions does each oracle control.
        for i in range(used_oracles):
            for j in range(N_times):
                this_loss = torch.sum(log_z_sm[1, :, assigned_dims[i] * N_times + j] * pref_all[i,:]) * mask_all[i,:] + \
                            torch.sum(log_z_sm[0, :, assigned_dims[i] * N_times + j] * (1-pref_all[i,:]) * mask_all[i,:])
                loss_unweighted += this_loss
        loss_unweighted = -loss_unweighted / (used_oracles * N_times)
    else: # old method
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
    return loss, loss_unweighted, used


loss_func = {
    'mse': compute_motion_mse,
    'kld': compute_z_kld,
    'sample': compute_sample_loss,
    'op': compute_oracle_preference_loss
}