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


from datasets import SequencesMotionDataset
from model.losses import get_observable_label

def plot_3d_traj(axs, traj):
        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

        axs.plot(x, y, z, label='Trajectory')
        axs.scatter(x[0], y[0], z[0], c='green', marker='o', s=100, label='Start')
        axs.scatter(x[-1], y[-1], z[-1], c='red', marker='o', s=100, label='End')

        axs.set_xlabel('X'); axs.set_ylabel('Y'); axs.set_zlabel('Z')
        axs.set_title('3D Trajectory')
        axs.legend()

        axs.set_box_aspect([1, 1, 1])


def compute_velocity_magnitude(velocities):
    """Compute the magnitude of velocity vectors"""
    return np.linalg.norm(velocities, axis=1)


def global_to_body_frame(vectors, orientations):
    """
    Convert any 3D vectors from global frame to body frame using quaternion orientations.
    
    Args:
        vectors: Array of shape (N, 3) with vectors in global frame (velocity, acceleration, etc.)
        orientations: Array of shape (N, 4) with quaternion orientations [w, x, y, z]
        
    Returns:
        Array of shape (N, 3) with vectors in body frame
    """
    return Rotation.from_quat(orientations).inv().apply(vectors)


def prepare_data_for_manifold(data_seqs, modes, sample_limit=1000):
    """
    Prepare data for manifold learning analysis by sampling and concatenating sequences.
    
    Args:
        data_seqs: Dictionary of sequences by mode
        modes: List of modes to include
        sample_limit: Maximum samples per sequence
        
    Returns:
        all_data: Concatenated data array
        labels: Array of mode labels
    """
    all_data = []
    labels = []
    
    for mode_idx, mode in enumerate(modes):
        for seq_name, data in data_seqs[mode].items():
            sample_rate = max(1, len(data) // sample_limit)
            data_ds = data[::sample_rate]
            
            all_data.append(data_ds)
            labels.extend([mode_idx] * len(data_ds))
    
    return np.concatenate(all_data), np.array(labels)


def prepare_aligned_data_for_manifold(data_seqs1, data_seqs2, modes, sample_limit=1000):
    """
    Prepare aligned data from two sequences for manifold learning analysis.
    
    Args:
        data_seqs1: Primary data sequences (e.g., IMU)
        data_seqs2: Secondary data sequences (e.g., velocity) 
        modes: List of modes to include
        sample_limit: Maximum samples per sequence
        
    Returns:
        all_data1: Concatenated primary data
        all_data2_mags: Magnitudes of secondary data
        labels: Array of mode labels
    """
    all_data1 = []
    all_data2_mags = []
    labels = []
    
    for mode_idx, mode in enumerate(modes):
        for seq_name in data_seqs1[mode]:
            if seq_name in data_seqs2[mode]:
                data1 = data_seqs1[mode][seq_name]
                data2 = data_seqs2[mode][seq_name]
                
                min_len = min(len(data1), len(data2))
                data1 = data1[:min_len]
                data2 = data2[:min_len]
                
                data2_magnitudes = np.linalg.norm(data2, axis=1)
                
                sample_rate = max(1, len(data1) // sample_limit)
                data1_ds = data1[::sample_rate]
                data2_mags_ds = data2_magnitudes[::sample_rate]
                
                all_data1.append(data1_ds)
                all_data2_mags.append(data2_mags_ds)
                labels.extend([mode_idx] * len(data1_ds))
    
    return np.vstack(all_data1), np.concatenate(all_data2_mags), np.array(labels)


def perform_manifold_analysis(data, method='tsne', apply_pca=False, pca_variance=0.95, random_state=42):
    """
    Perform manifold learning analysis with t-SNE or UMAP.
    
    Args:
        data: Input data array
        method: 'tsne' or 'umap'
        apply_pca: Whether to apply PCA first
        pca_variance: Variance to retain in PCA
        random_state: Random state for reproducibility
        
    Returns:
        manifold_result: 2D manifold coordinates
        pca_components: Number of PCA components (if applied)
    """
    processed_data = data
    pca_components = None
    
    if apply_pca:
        print("Applying PCA for dimension reduction...")
        pca = PCA(n_components=pca_variance)
        processed_data = pca.fit_transform(data)
        pca_components = processed_data.shape[1]
        print(f"Reduced dimensions from {data.shape[1]} to {pca_components}")
    
    print(f"Applying {method.upper()}...")
    if method == 'tsne':
        manifold = TSNE(n_components=2, random_state=random_state, 
                       perplexity=min(30, len(processed_data)//10))
    else:  # umap
        manifold = umap.UMAP(n_components=2, random_state=random_state,
                           n_neighbors=50)
    
    manifold_result = manifold.fit_transform(processed_data)
    return manifold_result, pca_components


def plot_manifold_scatter(manifold_coords, labels, color_values, modes, title, save_path, 
                         use_color_for_values=True, figsize=(12, 10)):
    """
    Unified function to plot manifold learning results with different visualization styles.
    
    Args:
        manifold_coords: Manifold coordinates
        labels: Mode labels (0, 1, ...)
        color_values: Values to use for coloring/transparency
        modes: Mode names
        title: Plot title
        save_path: Path to save figure
        use_color_for_values: If True, use colormap for values; if False, use transparency
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    if use_color_for_values:
        # Color-based visualization with distinct markers
        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
        markers = ['o', '^']
        edge_colors = ['white', 'black']
        
        for mode_idx, mode in enumerate(modes):
            mode_mask = labels == mode_idx
            scatter = plt.scatter(
                manifold_coords[mode_mask, 0], manifold_coords[mode_mask, 1],
                c=color_values[mode_mask], cmap='viridis', norm=norm,
                marker=markers[mode_idx], s=25, alpha=0.8,
                edgecolors=edge_colors[mode_idx], linewidths=0.5,
                label=mode
            )
        
        # Add colorbar and custom legend
        cbar = plt.colorbar(scatter)
        cbar.set_label('Velocity Magnitude (m/s)', rotation=270, labelpad=20)
        
        legend_elements = [
            plt.Line2D([0], [0], marker=markers[i], color='w', label=mode,
                      markerfacecolor='gray', markersize=8, 
                      markeredgecolor=edge_colors[i], markeredgewidth=0.5)
            for i, mode in enumerate(modes)
        ]
        plt.legend(handles=legend_elements, title="Dataset Type", loc='upper right')
        
    else:
        # Transparency-based visualization
        mode_colors = ['blue', 'green']
        
        # Calculate transparency values
        if color_values.max() > color_values.min():
            alpha_values = 0.2 + 0.7 * (color_values - color_values.min()) / (color_values.max() - color_values.min())
        else:
            alpha_values = np.full_like(color_values, 0.8)
        
        for mode_idx, mode in enumerate(modes):
            mode_mask = labels == mode_idx
            for i in np.where(mode_mask)[0]:
                plt.scatter(manifold_coords[i, 0], manifold_coords[i, 1], 
                           c=mode_colors[mode_idx], alpha=alpha_values[i], s=5)
        
        # Add legend and colorbar
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=mode,
                      markerfacecolor=color, markersize=10)
            for mode, color in zip(modes, mode_colors)
        ]
        plt.legend(handles=legend_elements, title="Dataset Type")
        
        sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(color_values.min(), color_values.max()), 
            cmap='Greys'
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Velocity Magnitude (m/s)')
    
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def analyze_and_plot_manifold(data_seqs, modes, frame_name, data_type="velocity", 
                             reference_seqs=None, apply_pca=False, method='tsne'):
    """
    Complete pipeline for manifold learning analysis and visualization.
    
    Args:
        data_seqs: Data sequences to analyze
        modes: List of mode names
        frame_name: Frame name for titles
        data_type: Type of data being analyzed
        reference_seqs: Reference sequences for color coding (optional)
        apply_pca: Whether to apply PCA preprocessing
        method: 'tsne' or 'umap'
    """
    print(f"Performing {method.upper()} on {data_type} data...")
    
    if reference_seqs is not None:
        # Aligned data analysis (e.g., IMU with velocity magnitudes)
        all_data, color_values, labels = prepare_aligned_data_for_manifold(
            data_seqs, reference_seqs, modes
        )
        use_color = True
    else:
        # Single data analysis (e.g., velocity vectors)
        all_data, labels = prepare_data_for_manifold(data_seqs, modes)
        color_values = np.linalg.norm(all_data, axis=1)
        use_color = False
    
    # Perform manifold learning
    manifold_coords, pca_dims = perform_manifold_analysis(all_data, method=method, apply_pca=apply_pca)
    
    # Generate title and filename
    title = f'{method.upper()} of {data_type.title()} in {frame_name}'
    if reference_seqs is not None:
        title += ' (Color = Velocity Magnitude)'
    else:
        title += '\n(Transparency = Magnitude)'
    
    filename = f'./evaluation/{method}_{data_type}_{frame_name.lower().replace("-", "_")}.png'
    
    # Plot results
    plot_manifold_scatter(manifold_coords, labels, color_values, modes, title, filename, 
                         use_color_for_values=use_color)
    
    if pca_dims is not None:
        print(f"PCA reduced dimensions to {pca_dims}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets/EuRoC/Euroc_body.conf",
        help="config file path, i.e., configs/Euroc.conf",
    )
    parser.add_argument(
        "--frame",
        type=str,
        choices=["glob", "body"],
        default="body",
        help="Use velocity in global or body frame (default: body)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["abs", "obs"],
        default="abs",
        help="Use absolute or observable velocity (default: abs)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tsne", "umap"],
        default="tsne",
        help="Manifold learning method (default: tsne)"
    )
    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.config)
    if args.type == "obs":
        args.frame = "body"

    train_dataset = SequencesMotionDataset(data_set_config=conf.train)
    test_dataset = SequencesMotionDataset(data_set_config=conf.test)
    eval_dataset = SequencesMotionDataset(data_set_config=conf.eval)

    ts_seqs = {'train': {}, 'test': {}, 'eval': {}}
    imu_seqs = {'train': {}, 'test': {}, 'eval': {}}
    pos_seqs = {'train': {}, 'test': {}, 'eval': {}}
    vel_seqs = {'train': {}, 'test': {}, 'eval': {}}
    rot_seqs = {'train': {}, 'test': {}, 'eval': {}}
    for mode, dataset, config in [
        ('train', train_dataset, conf.train),
        ('test', test_dataset, conf.test),
        ('eval', eval_dataset, conf.eval)
    ]:
        for i, seq_name in enumerate(config.data_list[0].data_drive):
            name = seq_name.split('/')[1] if config.data_list[0].name == 'BlackBird' else seq_name
            ts_seqs[mode][name] = dataset.ts[i]
            imu_seqs[mode][name] = np.concatenate((dataset.gyro[i], dataset.acc[i]), axis=1)
            pos_seqs[mode][name] = dataset.gt_pos[i]
            rot_seqs[mode][name] = dataset.gt_ori[i]
            if args.type == "obs":
                vel_seqs[mode][name] = get_observable_label(
                    dataset.ts[i][None, :, None], 
                    dataset.gt_ori[i][None, :], 
                    dataset.gt_velo[i][None, :]
                ).squeeze()
            else:
                vel_seqs[mode][name] = dataset.gt_velo[i]
    
    for mode in ['train', 'test', 'eval']:
        for name in vel_seqs[mode]:
            if name in rot_seqs[mode] and args.frame == "body":
                imu_seqs[mode][name][:, :3] = global_to_body_frame(
                    imu_seqs[mode][name][:, :3],
                    rot_seqs[mode][name][:-1]
                )
                imu_seqs[mode][name][:, 3:] = global_to_body_frame(
                    imu_seqs[mode][name][:, 3:],
                    rot_seqs[mode][name][:-1]
                )
                vel_seqs[mode][name] = global_to_body_frame(
                    vel_seqs[mode][name], 
                    rot_seqs[mode][name]
                ) if args.type == "abs" else vel_seqs[mode][name]
    
    plt.rcParams['font.sans-serif'] = 'Nimbus Sans'
    '''
    plt.figure(figsize=(20, 5))
    train_trajs = list(pos_seqs['train'].items())
    for i in range(min(5, len(train_trajs))):
        name, traj = train_trajs[i]
        ax = plt.subplot(1, 5, i+1, projection='3d')
        plot_3d_traj(ax, traj[:len(traj)])
        ax.set_title(f'Train: {name}')
    plt.tight_layout()
    plt.savefig('./evaluation/train_traj.png', dpi=300)
    '''
    
    frame_name = "Body-Frame" if args.frame == "body" else "Global-Frame"
    modes = ['train', 'eval']

    # Velocity manifold analysis
    analyze_and_plot_manifold(
        vel_seqs, modes, frame_name, 
        data_type="velocity",
        method=args.method
    )

    # IMU manifold analysis with velocity magnitude coloring
    analyze_and_plot_manifold(
        imu_seqs, modes, frame_name, 
        data_type="imu_features", 
        reference_seqs=vel_seqs, 
        apply_pca=True,
        method=args.method
    )