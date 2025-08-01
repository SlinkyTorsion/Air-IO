import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import argparse
import umap
import numpy as np
import matplotlib.pyplot as plt
from pyhocon import ConfigFactory
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.transform import Rotation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy import stats

from datasets import SequencesMotionDataset
from model.losses import get_observable_label


def global_to_body_frame(vectors, orientations):
    return Rotation.from_quat(orientations).inv().apply(vectors)


def quat_to_rpy_degrees(quaternions):
    """Convert quaternions to Roll-Pitch-Yaw angles in degrees with continuity wrapping."""
    rotations = Rotation.from_quat(quaternions)
    rpy_rad = rotations.as_euler('xyz', degrees=False)
    rpy_deg = np.degrees(rpy_rad)
    
    # Apply angle wrapping to maintain continuity in histograms
    for i in range(3):
        angles = rpy_deg[:, i]
        if np.min(angles) < -90 and np.max(angles) > 90:
            rpy_deg[:, i] = np.where(angles < 0, angles + 360, angles)
    
    return rpy_deg


def load_datasets(conf, data_type, transform=None):
    train_dataset = SequencesMotionDataset(data_set_config=conf.train)
    test_dataset = SequencesMotionDataset(data_set_config=conf.test)
    eval_dataset = SequencesMotionDataset(data_set_config=conf.eval)

    data_lists = {mode: {'ts': [], 'imu': [], 'pos': [], 'vel': [], 'rot': [], 'sequences': []} 
                  for mode in ['train', 'test', 'eval']}
    
    for mode, dataset, config in [
        ('train', train_dataset, conf.train),
        ('test', test_dataset, conf.test),
        ('eval', eval_dataset, conf.eval)
    ]:
        for i, seq in enumerate(config.data_list[0].data_drive):
            seq_data = {
                'name': seq,
                'ts': dataset.ts[i],
                'imu': np.concatenate((dataset.gyro[i], dataset.acc[i]), axis=1),
                'pos': dataset.gt_pos[i][:-1],
                'rot': dataset.gt_ori[i][:-1]
            }

            args.frame = config.coordinate
            
            if data_type == "obs":
                assert args.frame == 'body_coord'
                glob_velo = dataset.gt_ori[i] * dataset.gt_velo[i]
                seq_data['vel'] = get_observable_label(
                    dataset.ts[i][None, :, None], dataset.gt_ori[i][None, :], glob_velo[None, :]
                ).squeeze()
            else:
                seq_data['vel'] = dataset.gt_velo[i]

            if transform is not None:
                seq_data['imu'][:, :3] = np.dot(np.array(transform), seq_data['imu'][:, :3].T).T
                seq_data['imu'][:, 3:] = np.dot(np.array(transform), seq_data['imu'][:, 3:].T).T
                seq_data['rot'] = (Rotation.from_quat(seq_data['rot']) * Rotation.from_matrix(transform).inv()).as_quat()
                seq_data['vel'] = np.dot(np.array(transform), seq_data['vel'].T).T

            data_lists[mode]['sequences'].append(seq_data)
            for key in ['ts', 'imu', 'pos', 'rot', 'vel']:
                data_lists[mode][key].append(seq_data[key])

    for mode in data_lists:
        for key in ['ts', 'imu', 'pos', 'vel', 'rot']:
            data_lists[mode][key] = np.concatenate(data_lists[mode][key])

    return data_lists


def get_data_config(data_type, frame_name="Body-Frame"):
    """Get configuration for different data types."""
    configs = {
        'imu': {
            'sensors': [
                ('gyro', ['ωx (rad/s)', 'ωy (rad/s)', 'ωz (rad/s)'], slice(0, 3), 'Gyroscope', 0),
                ('acc', ['ax (m/s²)', 'ay (m/s²)', 'az (m/s²)'], slice(3, 6), 'Accelerometer', 1)
            ],
            'title': f'IMU Distribution Comparison ({frame_name})',
            'filename_suffix': 'imu_histogram_comparison'
        },
        'velocity': {
            'sensors': [
                ('vel', ['vx (m/s)', 'vy (m/s)', 'vz (m/s)'], slice(0, 3), 'Velocity', 0)
            ],
            'title': f'Velocity Distribution Comparison ({frame_name})',
            'filename_suffix': 'velocity_histogram_comparison'
        },
        'orientation': {
            'sensors': [
                ('rpy', ['Roll (°)', 'Pitch (°)', 'Yaw (°)'], slice(0, 3), 'Orientation (RPY)', 0)
            ],
            'title': f'Orientation Distribution Comparison ({frame_name})',
            'filename_suffix': 'orientation_histogram_comparison'
        }
    }
    return configs.get(data_type, {})


# ---------------- Histogram ---------------- # 
def compute_data_statistics(data, component_names):
    """Compute statistics for multi-dimensional data."""
    statistics = {}
    for i, component in enumerate(component_names):
        component_data = data[:, i]
        statistics[component] = {
            'mean': np.mean(component_data), 
            'std': np.std(component_data),
            'min': np.min(component_data), 
            'max': np.max(component_data)
        }
    return statistics


def plot_data_histogram(data_dict, data_type="imu", frame_name="Body-Frame", save_dir="./evaluation"):
    """General histogram plotting function for different data types."""
    config = get_data_config(data_type, frame_name)
    if not config:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    sensors = config['sensors']
    max_components = max(len(sensor[1]) for sensor in sensors)
    
    # Setup figure - rows for different sensor types, columns for components
    fig, axes = plt.subplots(len(sensors), max_components, figsize=(4*max_components, 4*len(sensors)))
    if len(sensors) == 1 and max_components == 1:
        axes = np.array([[axes]])
    elif len(sensors) == 1:
        axes = axes.reshape(1, -1)
    elif max_components == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(data_dict)))
    
    # Collect statistics for console output
    all_stats = {}
    for dataset_name, data in data_dict.items():
        all_stats[dataset_name] = {}
        for sensor_type, component_labels, indices, _, _ in sensors:
            sensor_data = data[:, indices]
            component_names = [label.split()[0] for label in component_labels]
            all_stats[dataset_name][sensor_type] = compute_data_statistics(sensor_data, component_names)
    
    # Plot histograms
    for sensor_idx, (sensor_type, component_labels, indices, title_prefix, row) in enumerate(sensors):
        for comp_idx, component_label in enumerate(component_labels):
            if len(sensors) == 1:
                ax = axes[0, comp_idx] if max_components > 1 else axes[0, 0]
            else:
                ax = axes[sensor_idx, comp_idx] if max_components > 1 else axes[sensor_idx, 0]
            
            for dataset_idx, (dataset_name, data) in enumerate(data_dict.items()):
                if indices.stop > data.shape[1]:
                    print(f"Warning: {dataset_name} data has {data.shape[1]} components, expected at least {indices.stop}")
                    continue
                    
                component_data = data[:, indices][:, comp_idx]
                
                # Plot histogram
                ax.hist(component_data, bins=50, alpha=0.6, density=True, 
                       label=dataset_name, color=colors[dataset_idx],
                       edgecolor='black', linewidth=0.5)
                
                # Add mean line
                mean_val = np.mean(component_data)
                ax.axvline(mean_val, color=colors[dataset_idx], 
                          linestyle='--', alpha=0.9, linewidth=2)
            
            ax.set_xlabel(component_label)
            ax.set_ylabel('Density')
            ax.set_title(f'{title_prefix} - {component_label.split()[0]}')
            ax.grid(True, alpha=0.3)
            
            if comp_idx == 0:  # Legend only on first column
                ax.legend(loc='upper right')
    
    # Hide unused subplots
    for sensor_idx in range(len(sensors)):
        for comp_idx in range(len(sensors[sensor_idx][1]), max_components):
            if len(sensors) == 1:
                ax = axes[0, comp_idx] if max_components > 1 else None
            else:
                ax = axes[sensor_idx, comp_idx] if max_components > 1 else None
            if ax:
                ax.set_visible(False)
    
    plt.suptitle(config['title'], fontsize=16)
    plt.tight_layout()
    
    # Save plot
    filename = f'{save_dir}/{config["filename_suffix"]}_{frame_name.lower().replace("-", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics summary to console
    print(f"\n{data_type.upper()} Statistics Summary ({frame_name}):")
    print("=" * 80)
    for dataset_name in all_stats.keys():
        print(f"\n{dataset_name}:")
        for sensor_type, component_labels, _, title_prefix, _ in sensors:
            print(f"  {title_prefix}:")
            for component, stats in all_stats[dataset_name][sensor_type].items():
                print(f"    {component}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}")
    
    print(f"\nHistogram saved to: {filename}")
    return all_stats


# ---------------- Manifold ---------------- # 
def prepare_data_for_manifold(inputs, reference_seqs=None):
    """Prepare data for manifold learning from dictionary inputs."""
    all_data = []
    labels = []
    color_values = []
    
    # Apply downsampling if data is too large
    for idx, (name, data) in enumerate(inputs.items()):
        max_samples = 10000
        if len(data) > max_samples:
            step = len(data) // max_samples
            data = data[::step]
            print(f"Downsampled {name} from {len(inputs[name])} to {len(data)} samples")
        
        all_data.append(data)
        labels.extend([idx] * len(data))

        if reference_seqs and name in reference_seqs:
            ref_data = reference_seqs[name]
            if len(inputs[name]) > max_samples:
                ref_data = ref_data[::step]
            color_values.extend(np.linalg.norm(ref_data, axis=1))
    
    return (np.concatenate(all_data), 
            np.array(labels), 
            np.array(color_values) if color_values else None)


def perform_manifold_analysis(data, method='tsne', apply_pca=False, random_state=42):
    """Apply PCA (optional) and manifold learning."""
    if apply_pca:
        pca = PCA(n_components=0.95)
        data = pca.fit_transform(data)
        print(f"PCA reduced dimensions to {data.shape[1]}")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=random_state, 
                      perplexity=min(30, len(data)//10))
    else:
        reducer = umap.UMAP(n_components=2, random_state=random_state)
    
    return reducer.fit_transform(data)


def plot_manifold_results(coords, labels, color_values, modes, title, save_path):
    """Plot manifold learning results with automatic styling."""
    plt.figure(figsize=(12, 10))
    
    if color_values is not None:
        # Use colormap with distinct markers for each dataset
        markers = ['o', '^', 's', 'D', 'v'][:len(modes)]
        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
        
        for i, mode in enumerate(modes):
            mask = labels == i
            scatter = plt.scatter(coords[mask, 0], coords[mask, 1],
                       c=color_values[mask], cmap='viridis', norm=norm,
                       marker=markers[i], s=20, alpha=0.7, label=mode,
                       facecolors='none', linewidths=1)
        
        plt.colorbar(label='Velocity Magnitude (m/s)')
        plt.legend(title="Dataset")
    else:
        # Simple color-coded scatter with hollow markers
        colors = plt.cm.Set1(np.linspace(0, 1, len(modes)))
        for i, (mode, color) in enumerate(zip(modes, colors)):
            mask = labels == i
            plt.scatter(coords[mask, 0], coords[mask, 1],
                       cmap='viridis', edgecolors='k', label=mode, 
                       alpha=0.8, s=30, linewidths=0.5)
        plt.legend()
    
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_and_plot_manifold(data_dict, frame_name, data_type="features", 
                             reference_dict=None, apply_pca=False, method='tsne'):
    """Complete manifold analysis pipeline."""
    print(f"Analyzing {data_type} with {method.upper()}...")
    
    # Prepare data
    data, labels, colors = prepare_data_for_manifold(data_dict, reference_dict)
    
    # Apply manifold learning
    coords = perform_manifold_analysis(data, method=method, apply_pca=apply_pca)
    
    # Create plot
    modes = list(data_dict.keys())
    title = f'{method.upper()} Analysis: {data_type.title()} ({frame_name})'
    filename = f'./evaluation/{method}_{data_type}_{frame_name.lower().replace("-", "_")}.png'
    
    plot_manifold_results(coords, labels, colors, modes, title, filename)
    print(f"Saved plot to {filename}")


def plot_trajectory_with_orientation(ax, pos_traj, rot_traj, title="Trajectory", orientation_interval=50):
    """Plot 3D trajectory with orientation frames."""
    if hasattr(pos_traj, 'detach'):
        pos_traj = pos_traj.detach().cpu().numpy()
    if hasattr(rot_traj, 'detach'):
        rot_traj = rot_traj.detach().cpu().numpy()
    
    x, y, z = pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2]
    
    # Plot trajectory
    ax.plot(x, y, z, color='purple', linewidth=1.5)
    ax.scatter(x[0], y[0], z[0], c='green', s=60, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=60, label='End')
    
    # Plot orientation frames
    indices = range(0, len(pos_traj), orientation_interval)
    trajectory_range = max(np.ptp(x), np.ptp(y), np.ptp(z))
    scale = max(0.5, trajectory_range * 0.1)
    
    # Standard basis vectors and colors
    basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = ['red', 'green', 'blue']
    
    for idx, i in enumerate(indices):
        if i >= len(pos_traj):
            break
            
        pos = pos_traj[i]
        quat = rot_traj[i]
        alpha = 0.2 + 0.4 * (idx / (len(indices) - 1))
        
        rotation = Rotation.from_quat(quat)
        rotated_axes = rotation.apply(basis)

        for axis, color in zip(rotated_axes, colors):
            end_point = pos + axis * scale
            ax.plot([pos[0], end_point[0]], [pos[1], end_point[1]], [pos[2], end_point[2]], 
                   color=color, linewidth=6, alpha=alpha, solid_capstyle='round')
            ax.scatter(*end_point, c=color, s=30, alpha=alpha)
    
    max_range = max(np.ptp(x), np.ptp(y), np.ptp(z))
    center = [np.mean([np.min(coord), np.max(coord)]) for coord in [x, y, z]]
    
    for i, (axis_func, label) in enumerate([(ax.set_xlim, 'X (m)'), (ax.set_ylim, 'Y (m)'), (ax.set_zlim, 'Z (m)')]):
        axis_func(center[i] - max_range/2, center[i] + max_range/2)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.3)


def visualize_sequences(datasets, dataset_name, mode):
    """Visualize sequences from datasets with position and orientation."""
    dataset = datasets[dataset_name]
    sequences = dataset[mode]['sequences']
    
    if not sequences:
        print(f"No sequences found for {dataset_name} {mode}")
        return
    
    num_sequences = len(sequences)
    plt.figure(figsize=(4*num_sequences, 4))
    
    for i, seq in enumerate(sequences):
        ax = plt.subplot(1, num_sequences, i + 1, projection='3d')
        plot_trajectory_with_orientation(
            ax, seq['pos'], seq['rot'],
            title=f"{seq['name']}\n(with orientation)",
            orientation_interval=len(seq['pos']) // 3
        )
    
    plt.tight_layout()
    save_path = f'./evaluation/{dataset_name}_{mode}_trajectories.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Trajectory visualization saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze datasets")
    parser.add_argument("--config", type=str, nargs='+', default=["configs/datasets/EuRoC/Euroc_body.conf"])
    parser.add_argument("--type", type=str, choices=["abs", "obs"], default="abs")
    parser.add_argument("--method", type=str, choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--histogram",action="store_true")
    parser.add_argument("--manifold",action="store_true")
    args = parser.parse_args()
    plt.rcParams['font.sans-serif'] = 'Nimbus Sans'

    datasets = {}
    coordinate_frames = set()

    # tranform EuRoC to BlackBird
    transform = [[0, 1, 0],
                 [0, 0, -1],
                 [-1, 0, 0]]
    
    for config_path in args.config:
        name = config_path.split('/')[2]
        print(f"Loading dataset {name} from {config_path}...")
        conf = ConfigFactory.parse_file(config_path)
        coordinate_frames.add(conf.train.coordinate)
        if len(coordinate_frames) > 1:
            raise ValueError(f"Inconsistent coordinate frames: {coordinate_frames}")
        
        if name == 'BlackBird':
            transform = None
        datasets[name] = load_datasets(conf, args.type, transform)
        for mode in ['test']:
            visualize_sequences(datasets, name, mode)
    
    frame_name = "Body-Frame" if args.frame == "body_coord" else "Global-Frame"

    imu_data = {}
    vel_data = {}
    rot_data = {}
    for name in datasets:
        imu_data[name] = np.concatenate([values['imu'] for _, values in datasets[name].items()])
        vel_data[name] = np.concatenate([values['vel'] for _, values in datasets[name].items()])

        quat_data = np.concatenate([values['rot'] for _, values in datasets[name].items()])
        rot_data[name] = quat_to_rpy_degrees(quat_data)

    # Histogram analysis (optional)
    if args.histogram:
        print(f"\nAnalyzing data distributions ({frame_name})...")
        imu_stats = plot_data_histogram(imu_data, data_type="imu", frame_name=frame_name)
        vel_stats = plot_data_histogram(vel_data, data_type="velocity", frame_name=frame_name)
        rot_stats = plot_data_histogram(rot_data, data_type="orientation", frame_name="Global-Frame")

    # Manifold analysis (optional)
    if args.manifold:
        # Velocity manifold analysis
        analyze_and_plot_manifold(
            vel_data, frame_name, 
            data_type="velocity",
            reference_dict=vel_data,
            method=args.method
        )

        # IMU manifold analysis with velocity magnitude coloring
        analyze_and_plot_manifold(
            imu_data, frame_name,
            data_type="imu_features", 
            reference_dict=vel_data, 
            apply_pca=True,
            method=args.method
        )