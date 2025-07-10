import os
import sys
import time
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Nimbus Sans', 'DejaVu Sans', 'Arial', 'sans-serif']

import torch.utils.data as Data
import argparse
import pickle

import tqdm
from utils import move_to, save_state
from pyhocon import ConfigFactory

from datasets import collate_fcs, SequencesMotionDataset
from model import net_dict
from utils import *
from model.losses import get_observable_label, retrieve_from_obser


def plot_velocity_comparison(axs, name, mode, ts, inf_state, label):
    import matplotlib.pyplot as plt
    
    if inf_state is None:
        return
    inf_state = inf_state[0].cpu().numpy()
    label = label[0].cpu().numpy()
    if mode != 'absolute':
        ts_values = ts.flatten()[:-1].cpu().numpy()
    else:
        ts_values = ts.flatten().cpu().numpy()
    
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    components = ['x', 'y', 'z']
    
    for i in range(3):
        axs[i].plot(ts_values, inf_state[:, i], label='Predicted')
        axs[i].plot(ts_values, label[:, i], label='Ground Truth')
        axs[i].set_ylabel(f'Velocity {components[i]}')
        axs[i].legend()
        axs[i].grid(True, linestyle='--', alpha=0.7)
    axs[0].set_title(f'{mode} velocity')
    axs[2].set_xlabel('Time (s)')
    return axs

def inference(network, loader, confs, coord):
    '''
    Correction inference
    save the corrections generated from the network.
    '''
    network.eval()
    evaluate_states = {}
    with torch.no_grad():
        inte_state = None
        for data, _, label in tqdm.tqdm(loader):
            data, label = move_to([data, label],  confs.device)
            rot = label['gt_rot'][:,:-1,:].Log().tensor()
            inte_state = network.forward(data, rot, confs.obsersup)
            
            ts = network.get_label(data['ts'][...,None])
            rot = network.get_label(label['gt_rot'])
            gt_label = network.get_label(label['gt_vel'])
            rel_label = torch.diff(gt_label, axis=1)
            
            obs_state = inte_state['net_vel'] if confs.obsersup else None
            obser_label = get_observable_label(ts, rot, gt_label)
            
            if coord == 'glob_coord':
                if confs.obsersup:
                    init_state = gt_label[0][0]
                    rel_state, abs_state = retrieve_from_obser(ts, rot, obs_state, init_state)
                else:
                    rel_state, abs_state = torch.diff(inte_state['net_vel'], axis=1), inte_state['net_vel']
                    obs_state = get_observable_label(ts, rot, abs_state)
            elif coord == 'body_coord':
                if confs.obsersup:
                    init_state = gt_label[0][0]
                    rel_state, abs_state = retrieve_from_obser(ts, rot, obs_state, init_state)
                else:
                    abs_state = rot * inte_state['net_vel']
                    rel_state, obs_state = torch.diff(abs_state, axis=1), get_observable_label(ts, rot, abs_state)

            fig, axs = plt.subplots(3, 3, figsize=(15, 9))
            ave = (abs_state - gt_label).norm(dim=-1).mean().item()

            plot_velocity_comparison(axs[:, 0], dataset_name, 'observable', ts, obs_state, obser_label)
            plot_velocity_comparison(axs[:, 1], dataset_name, 'relative', ts, rel_state, rel_label)
            plot_velocity_comparison(axs[:, 2], dataset_name, 'absolute', ts, abs_state, gt_label)

            plt.suptitle(f'Sequence: {dataset_name} - AVE: {ave:.4f} m/s', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_folder}/{dataset_name}_velocity_comparison.png', dpi=300)
            plt.close(fig)
            
            inte_state['ts'] = network.get_label(data['ts'][...,None])[0]
            save_state(evaluate_states, inte_state)
           
        for k, v in evaluate_states.items():    
            evaluate_states[k] = torch.cat(v,  dim=-2)
    return evaluate_states

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/EuRoC/motion_body.conf', help='config file path')
    parser.add_argument('--load', type=str, default=None, help='path for specific model check point, Default is the best model')
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
    parser.add_argument('--seqlen', type=int, default=1000, help='window size.')
    parser.add_argument('--whole', default=True, action="store_true", help='estimate the whole seq')


    args = parser.parse_args(); print(args)
    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = args.device
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)
    conf['device'] = args.device
    dataset_conf = conf.dataset.inference
    
    
    network = net_dict[conf.train.network](conf.train).to(args.device).double()
    save_folder = os.path.join(conf.general.exp_dir, "evaluate")
    os.makedirs(save_folder, exist_ok=True)

    if args.load is None:
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt")
    else:
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt", args.load)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device),weights_only=True)
        print("loaded state dict %s in epoch %i"%(ckpt_path, checkpoint["epoch"]))
        network.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise KeyError(f"No model loaded {ckpt_path}")
        sys.exit()
        
    if 'collate' in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate.type]
    else:
        collate_fn = collate_fcs['base']

    # Body alignment
    transform = {}
    exp_dataset_name = dataset_conf.data_list[0].name
    net_dataset_name = conf.general.exp_dir.split('/')[1]
    if dataset_conf.coordinate == 'body_coord':
        if exp_dataset_name == 'Euroc' and net_dataset_name != exp_dataset_name:
            print("Performing body coordinate alignment. (EuRoC -> BlackBird)")
            transform['euroc'] = [[0, 1, 0],
                                      [0, 0, -1],
                                      [-1, 0, 0]]
        elif exp_dataset_name == 'BlackBird' and net_dataset_name == 'euroc':
            print("Performing body coordinate alignment. (BlackBird -> EuRoC)")
            transform['blackbird'] = [[0, 0, -1],
                                          [1, 0, 0],
                                          [0, -1, 0]]
        else:
            transform[exp_dataset_name.lower()] = None

    cov_result, rmse = [], []
    net_out_result = {}
    evals = {}
    dataset_conf.data_list[0]["window_size"] = args.seqlen
    dataset_conf.data_list[0]["step_size"] = args.seqlen
    for data_conf in dataset_conf.data_list:
        for path in data_conf.data_drive:
            if args.whole:
                dataset_conf["mode"] = "inference"
            else:
                dataset_conf["mode"] = "infevaluate"
            dataset_conf["exp_dir"] = conf.general.exp_dir
            dataset_name = path.split("/")[1] if data_conf.name == 'BlackBird' else path
            print("dataset_name:", dataset_name)
            
            eval_dataset = SequencesMotionDataset(data_set_config=dataset_conf, data_path=path, data_root=data_conf["data_root"],
                body_transform=transform
            )
            eval_loader = Data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, 
                                            shuffle=False, collate_fn=collate_fn, drop_last = False)
            inference_state = inference(network=network, loader = eval_loader, confs=conf.train, coord=dataset_conf.coordinate)  
            if not "cov" in inference_state.keys():
                    inference_state["cov"] = torch.zeros_like(inference_state["net_vel"])         
            inference_state['ts'] = inference_state['ts']
            inference_state['net_vel'] = inference_state['net_vel'][0] #TODO: batch size != 1
            net_out_result[path] = inference_state

    net_result_path = os.path.join(conf.general.exp_dir, 'net_output.pickle')
    print("save netout, ", net_result_path)
    with open(net_result_path, 'wb') as handle:
        pickle.dump(net_out_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

