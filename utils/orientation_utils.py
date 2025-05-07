import numpy as np
import matplotlib.pyplot as plt
import pypose as pp
from scipy.spatial.transform import Rotation, Slerp
import os

def plot_orientation_comparison(airimu_rot, inte_rot, gt_rot, gt_time=None, state_time=None, 
                                data_name=None, show_plot=True):
    """
    Visualizes and compares different orientation data sources.
    
    Args:
        airimu_rot: AirIMU rotation data
        inte_rot: Integrated rotation data
        gt_rot: Ground truth rotation data
        gt_time: Time stamps for ground truth data
        state_time: Time stamps from state dictionary
        data_name: Name of dataset for title/saving
        save_path: Directory to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        None: Shows and/or saves the plot
    """
    
    try:
        airimu_euler = airimu_rot.euler().cpu().numpy()
        inte_euler = inte_rot.euler().cpu().numpy()
        gt_euler = gt_rot.euler().cpu().numpy()
    except (AttributeError, TypeError):
        airimu_euler = pp.SO3(airimu_rot).euler().cpu().numpy()
        inte_euler = pp.SO3(inte_rot).euler().cpu().numpy()
        if not isinstance(gt_rot, pp.SO3):
            gt_euler = pp.SO3(gt_rot).euler().cpu().numpy()
        else:
            gt_euler = gt_rot.euler().cpu().numpy()
    

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    time_axis = state_time if state_time is not None else np.arange(len(airimu_euler))
    

    if gt_time is not None and len(gt_euler) != len(time_axis):
        if len(gt_time) != len(time_axis):
            print(f"Resampling GT orientation from {len(gt_time)} to {len(time_axis)} points")
            gt_rot_scipy = Rotation.from_quat(gt_rot.data().cpu().numpy())
            
            valid_idx = (time_axis >= gt_time.min()) & (time_axis <= gt_time.max())
            valid_times = time_axis[valid_idx]
            
            if len(valid_times) > 0:
                interpolated_rot = Slerp(gt_time, gt_rot_scipy)(valid_times)
                gt_euler = np.zeros((len(time_axis), 3))
                gt_euler[valid_idx] = interpolated_rot.as_euler('xyz')
            else:
                print("Warning: No valid interpolation points")
                gt_euler = np.zeros_like(airimu_euler)
    
    axes[0].plot(time_axis, np.rad2deg(airimu_euler[:, 0]), 'r-', label='AirIMU')
    axes[0].plot(time_axis, np.rad2deg(inte_euler[:, 0]), 'b--', label='Integration')
    axes[0].plot(time_axis, np.rad2deg(gt_euler[:, 0]), 'g-.', label='Ground Truth')
    axes[0].set_ylabel('Roll (degrees)')
    axes[0].legend()
    
    axes[1].plot(time_axis, np.rad2deg(airimu_euler[:, 1]), 'r-', label='AirIMU')
    axes[1].plot(time_axis, np.rad2deg(inte_euler[:, 1]), 'b--', label='Integration')
    axes[1].plot(time_axis, np.rad2deg(gt_euler[:, 1]), 'g-.', label='Ground Truth')
    axes[1].set_ylabel('Pitch (degrees)')
    
    axes[2].plot(time_axis, np.rad2deg(airimu_euler[:, 2]), 'r-', label='AirIMU')
    axes[2].plot(time_axis, np.rad2deg(inte_euler[:, 2]), 'b--', label='Integration')
    axes[2].plot(time_axis, np.rad2deg(gt_euler[:, 2]), 'g-.', label='Ground Truth')
    axes[2].set_ylabel('Yaw (degrees)')
    axes[2].set_xlabel('Time')
    
    title = f'Comparison between AirIMU, Integration, and Ground Truth Rotations'
    if data_name:
        title += f' - {data_name}'
    plt.suptitle(title)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        
    return fig
