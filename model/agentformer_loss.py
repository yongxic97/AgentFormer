from eval import compute_ADE, compute_FDE, align_gt
import torch
from torch import nn


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

    z = data['oracle_eval_z']                                               # z   : [z_dim]; [bs, nz]
    pred = data['oracle_eval_dec_motion']                                   # pred: [z_dim]; [bs, fut_len, 2]; 
    gt = data['fut_motion_orig']                                            # gt  : [bs, fut_len, 2]
    
    pref_ade = oracle_prefers_smaller_de(z, pred, gt, compute_ADE)          # [bs]
    pref_fde = oracle_prefers_smaller_de(z, pred, gt, compute_FDE)          # [bs]
  
    z_tensor = torch.stack((z[0], z[1]))                                    # [z_dim, bs, nz]
    pref_all = torch.stack([pref_ade, pref_fde], dim=0).to(z_tensor.device) # [2, bs]
    # log_z_sm = torch.log(nn.functional.softmax(torch.log(z_tensor), dim=0))
    log_z_sm = torch.log(nn.functional.softmax(z_tensor, dim=0))
    # log_z_sm = nn.functional.softmax(z_tensor,dim=0)

    # In this case, we use two metrics, and z_dim is essentially two as well (for now).
    # Hence, we use dims 0-9 for ADE and 10-19 for FDE for better usage of prefernce.
    log_z_ade = log_z_sm[:,:,0:1]
    log_z_fde = log_z_sm[:,:,1:2]

    bp_loss_ade, bp_loss_fde = 0, 0

    N_times = 1
    for i in range(N_times):
        bp_loss_ade += (torch.sum(log_z_ade[1,:,i] * pref_all[0,:])
                        + torch.sum(log_z_ade[0,:,i] * (1-pref_all[0,:])))
        bp_loss_fde += (torch.sum(log_z_fde[1,:,i] * (1-pref_all[1,:]))
                        + torch.sum(log_z_fde[0,:,i] * pref_all[1,:]))
        # bp_loss_fde += (torch.sum(log_z_fde[1,:,i] * pref_all[1,:])
        #                 + torch.sum(log_z_fde[0,:,i] * (1-pref_all[1,:])))
    loss_unweighted = -(bp_loss_ade + bp_loss_fde) / N_times

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