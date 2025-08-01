import os

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pypose as pp
import torch

def visualize_motion(save_prefix, save_folder, outstate,infstate,label="AirIO"):
    ### visualize gt&netoutput velocity, 2d trajectory. 
    gt_x, gt_y, gt_z                = torch.split(outstate["poses_gt"][0].cpu(), 1, dim=1)
    airTraj_x, airTraj_y, airTraj_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)
    
    v_gt_x, v_gt_y, v_gt_z       = torch.split(outstate['vel_gt'][0][::50,:].cpu(), 1, dim=1)
    airVel_x, airVel_y, airVel_z = torch.split(infstate['net_vel'][0][::50,:].cpu(), 1, dim=1)
    
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(3, 2) 

    ax1 = fig.add_subplot(gs[:, 0]) 
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax3 = fig.add_subplot(gs[1, 1]) 
    ax4 = fig.add_subplot(gs[2, 1]) 
   
    #visualize traj 
    ax1.plot(airTraj_x, airTraj_y, label=label)
    ax1.plot(gt_x     , gt_y     , label="Ground Truth")
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.legend()
    
    #visualize vel
    ax2.plot(airVel_x,label=label)
    ax2.plot(v_gt_x,label="Ground Truth")
    
    ax3.plot(airVel_y,label=label)
    ax3.plot(v_gt_y,label="Ground Truth")
    
    ax4.plot(airVel_z,label=label)
    ax4.plot(v_gt_z,label="Ground Truth")
    
    ax2.set_xlabel('time')
    ax2.set_ylabel('velocity')
    ax2.legend()
    ax3.legend()
    ax4.legend()
    save_prefix += "_state.png"
    plt.savefig(os.path.join(save_folder, save_prefix), dpi = 300)
    plt.close()

def visualize_rotations(save_prefix, gt_rot, out_rot, inf_rot=None, save_folder=None):
    gt_euler = np.unwrap(pp.SO3(gt_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi
    outstate_euler = np.unwrap(pp.SO3(out_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi

    roe_raw = (pp.SO3(gt_rot).Inv() @ pp.SO3(out_rot)).Log().norm(dim=-1).cpu().numpy() * 180.0 / np.pi
    print(f"ROE - Raw Integration: {np.mean(roe_raw):.3f} degrees")
    
    legend_list = ["roll", "pitch","yaw"]
    fig, axs = plt.subplots(3)
    fig.suptitle("integrated orientation")
    for i in range(3):
        axs[i].plot(outstate_euler[:, i], color="b", linewidth=0.9)
        axs[i].plot(gt_euler[:, i], color="mediumseagreen", linewidth=0.9)
        axs[i].legend(["raw_" + legend_list[i], "gt_" + legend_list[i]])
        axs[i].grid(True)

    if inf_rot is not None:
        infstate_euler = np.unwrap(pp.SO3(inf_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi

        roe_airimu = (pp.SO3(gt_rot).Inv() @ pp.SO3(inf_rot)).Log().norm(dim=-1).cpu().numpy() * 180.0 / np.pi
        print(f"ROE - AirIMU: {np.mean(roe_airimu):.3f} degrees")
        print(f"ROE Improvement: {np.mean(roe_raw) - np.mean(roe_airimu):.3f} degrees")
        
        for i in range(3):
            axs[i].plot(infstate_euler[:, i], color="red", linewidth=0.9)
            axs[i].legend(["raw_" + legend_list[i], "gt_" + legend_list[i], "AirIMU_" + legend_list[i]])

        visualize_orientation_error(save_prefix, gt_euler, outstate_euler, infstate_euler, save_folder)
    else:
        visualize_orientation_error(save_prefix, gt_euler, outstate_euler, None, save_folder)
    
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, save_prefix + "_orientation_compare.png"), dpi=300)
    plt.close()

def visualize_orientation_error(save_prefix, gt_euler, out_euler, inf_euler=None, save_folder=None):
    legend_list = ["roll", "pitch", "yaw"]
    fig, axs = plt.subplots(3, figsize=(10, 8))
    fig.suptitle("orientation error")
    
    for i in range(3):
        axs[i].plot(out_euler[:, i] - gt_euler[:, i], color="b", linewidth=0.9)
        axs[i].set_ylabel(f"{legend_list[i]}")
        axs[i].grid(True)
        
        if inf_euler is not None:
            axs[i].plot(inf_euler[:, i] - gt_euler[:, i], color="red", linewidth=0.9)
            axs[i].legend(["raw_error", "AirIMU_error"])
        else:
            axs[i].legend(["raw_error"])
    
    axs[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, save_prefix + "_orientation_error.png"), dpi=300)
    plt.close()

def visualize_positions(save_prefix, gt_pos, out_pos, inf_pos=None, save_folder=None):
    gt_pos = gt_pos.cpu().numpy()
    out_pos = out_pos.cpu().numpy()

    legend_list = ['X', 'Y', 'Z']
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(3, 2) 

    ax1 = fig.add_subplot(gs[:, 0]) 
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax3 = fig.add_subplot(gs[1, 1]) 
    ax4 = fig.add_subplot(gs[2, 1])
    axs = [ax1, ax2, ax3, ax4] 

    fig.suptitle("integrated position")
    axs[0].plot(out_pos[:, 0], out_pos[:, 1], color="b", linewidth=0.9)
    axs[0].plot(gt_pos[:, 0], gt_pos[:, 1], color="mediumseagreen", linewidth=0.9)
    axs[0].legend(["raw", "gt"])
    axs[0].grid(True)
    for i in range(1, 4):
        axs[i].plot(out_pos[:, i-1], color="b", linewidth=0.9)
        axs[i].plot(gt_pos[:, i-1], color="mediumseagreen", linewidth=0.9)
        axs[i].legend(["raw_" + legend_list[i-1], "gt_" + legend_list[i-1]])
        axs[i].grid(True)
    
    if inf_pos is not None:
        inf_pos = inf_pos.cpu().numpy()
        axs[0].plot(inf_pos[:, 0], inf_pos[:, 1], color="red", linewidth=0.9)
        axs[0].legend(["raw", "gt", "AirIMU"])
        axs[0].grid(True)
        for i in range(1, 4):
            axs[i].plot(inf_pos[:, i-1], color="red", linewidth=0.9)
            axs[i].legend(
                [
                    "raw_" + legend_list[i-1],
                    "gt_" + legend_list[i-1],
                    "AirIMU_" + legend_list[i-1],
                ]
            )
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, save_prefix + "_position_compare.png"), dpi=300
        )
    plt.show()
    plt.close()

def visualize_ekf_result(save_prefix, save_folder, ekf_result, gtpos, gtvel, interp_net_vel=None):
    """
    Visualize EKF trajectory and velocity results compared to ground truth.
    
    Args:
        save_prefix (str): Prefix for saving the output file.
        save_folder (str): Folder to save the output file.
        ekf_result (numpy.ndarray): EKF state estimates containing positions at indices 6:9 and velocities at indices 3:6.
        gtpos (torch.Tensor): Ground truth positions.
        gtvel (torch.Tensor): Ground truth velocities.
        interp_net_vel (numpy.ndarray, optional): Interpolated network velocities.
    """
    # Convert torch tensors to numpy if needed
    if isinstance(gtpos, torch.Tensor):
        gtpos = gtpos.cpu().numpy()
    if isinstance(gtvel, torch.Tensor):
        gtvel = gtvel.cpu().numpy()
    
    # Extract positions and velocities from EKF result
    ekf_pos = ekf_result[:, 6:9]
    ekf_vel = ekf_result[:, 3:6]
    
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(3, 2)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[:, 0]) 
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax3 = fig.add_subplot(gs[1, 1]) 
    ax4 = fig.add_subplot(gs[2, 1]) 
    
    # Plot trajectories on 2D plane
    ax1.plot(ekf_pos[:, 0], ekf_pos[:, 1], label="EKF")
    ax1.plot(gtpos[:, 0], gtpos[:, 1], label="Ground Truth")
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.legend()
    ax1.grid(True)
    
    # Plot velocity components
    velocity_axes = [ax2, ax3, ax4]
    velocity_labels = ['X velocity', 'Y velocity', 'Z velocity']
    sample_indices = np.arange(0, len(ekf_vel), 50)
    
    for i, ax in enumerate(velocity_axes):
        ax.plot(sample_indices, ekf_vel[sample_indices, i], label="EKF")
        ax.plot(sample_indices, gtvel[sample_indices, i], label="Ground Truth")
        if interp_net_vel is not None:
            ax.plot(sample_indices, interp_net_vel[sample_indices, i], label="AirIO")
        ax.set_ylabel(velocity_labels[i])
        ax.legend()
        ax.grid(True)
    
    velocity_axes[-1].set_xlabel('Sample Index')
    
    plt.tight_layout()
    save_path = os.path.join(save_folder, f"{save_prefix}_ekf_state.png")
    plt.savefig(save_path, dpi=300)
    plt.close()