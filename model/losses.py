import torch
from .loss_func import loss_fc_list, diag_ln_cov_loss

def compute_multi_scale_components(rot, net_vel, label):
    assert rot.shape == net_vel.shape + 1
    omegas = rot[:, :-1, :].Inv() @ rot[:, 1:, :]
    omegas = omegas[:, ::2, :] @ omegas[:, 1::2, :]
    net_vel = net_vel[:, ::2, :] + omegas @ net_vel[:, 1::2, :]
    label = label[:, ::2, :] + omegas @ label[:, 1::2, :]
    return net_vel, label

def get_observable_label(ts, rot, label, coord):
    assert all(x.ndim == 3 for x in [ts, rot, label])

    label = rot * label if coord == 'body_coord' else label
    gravity = torch.tensor([0., 0., -9.81007], device=label.device)
    dt = torch.diff(ts, axis=1)

    obs_label = rot[:, :-1, :].Inv() * torch.diff(label, axis=1)
    return obs_label

def retrieve_from_obser(ts, rot, obs_state, init_state, coord):
    gravity = torch.tensor([0., 0., -9.81007]).to(obs_state.device)
    init_state = init_state[None, None, ...] if init_state.ndim == 1 else init_state
    init_state = rot[:, 0, :].unsqueeze(1) * init_state if coord == 'body_coord' else init_state
    
    rel_state = rot[:, :-1, :] * obs_state
    abs_state = torch.cumsum(rel_state, dim=1) + init_state
    abs_state = torch.cat((init_state, abs_state), dim=1)
    if coord == 'body_coord':
        abs_state = rot.Inv() * abs_state
        rel_state = rot[:, :-1, :].Inv() * rel_state
    return rel_state, abs_state

def motion_loss_(fc, pred, targ):
    dist = pred - targ
    loss = fc(dist)
    return loss, dist

def get_motion_loss(inte_state, label, confs, ts=None, rot=None):
    ## The state loss for evaluation
    loss, cov_loss = 0, {}
    loss_fc = loss_fc_list[confs.loss]
    
    obs_label = get_observable_label(ts, rot, label, confs.coord)
    if confs.obsersup:
        assert ts.shape[1] == label.shape[1]
        obs_vel = inte_state['net_vel']
        _, vel = retrieve_from_obser(ts, rot, obs_vel, label[:, :1, :], confs.coord)
    else:
        vel = inte_state['net_vel']
        obs_vel = get_observable_label(ts, rot, vel, confs.coord)

    vel_loss, vel_dist = motion_loss_(loss_fc, vel, label)
    obs_vel_loss, obs_vel_dist = motion_loss_(loss_fc, obs_vel, obs_label)
    abs_loss, obs_loss = confs.weight * vel_loss, confs.obs_weight * obs_vel_loss

    # Apply the covariance loss
    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()

        if "covaug" in confs and confs["covaug"] is True:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist, cov)
        else:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist.detach(), cov)
    loss = obs_loss if confs.obsersup else abs_loss
    return {'loss':loss, 'abs_loss': abs_loss, 'obs_loss': obs_loss, 'cov_loss':cov_loss}


def get_motion_RMSE(inte_state, label, confs, ts=None, rot=None):
    '''
    get the RMSE of the last state in one segment
    '''
    def _RMSE(x):
        return torch.sqrt((x.norm(dim=-1)**2).mean())
    cov_loss = 0

    obs_label = get_observable_label(ts, rot, label, confs.coord)
    if confs.obsersup:
        assert ts.shape[1] == label.shape[1]
        obs_vel = inte_state['net_vel']
        _, abs_vel = retrieve_from_obser(ts, rot, obs_vel, label[:, :1, :], confs.coord)
    else:
        abs_vel = inte_state['net_vel']
        obs_vel = get_observable_label(ts, rot, abs_vel, confs.coord)
    
    abs_dist, obs_dist = (abs_vel - label), (obs_vel - obs_label)
    abs_dist, obs_dist = torch.mean(abs_dist,dim=-2), torch.mean(obs_dist,dim=-2)
    abs_loss, obs_loss = _RMSE(abs_dist)[None,...], _RMSE(obs_dist)[None,...]
    loss = abs_loss + obs_loss

    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()
    
    return {'loss': loss,
            'abs_loss': abs_loss,
            'obs_loss': obs_loss,
            'abs_dist': abs_dist.norm(dim=-1).mean(),
            'obs_dist': obs_dist.norm(dim=-1).mean(),
            'cov_loss': cov_loss}
