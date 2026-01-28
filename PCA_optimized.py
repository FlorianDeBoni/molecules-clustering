#!/usr/bin/env python3

import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import IncrementalPCA
import time
import ijson
import sys

# On ne lit pas tout le JSON pour eviter le crash
def get_metadata(filename):
    
    with open(filename, 'rb') as f:
        parser = ijson.items(f, 'metadata')
        metadata = next(parser)
    
    print("Snapshots: ",{metadata['n_frames']})
    print("Atoms: ",{metadata['n_atoms']})
    print("Clusters:", {metadata['n_clusters']})
    
    return metadata

# On recupere les centroids
def get_centroids(filename):
    with open(filename, 'rb') as f:
        parser = ijson.items(f, 'centroids')
        centroids = next(parser)
    return centroids


def stream_snapshots_to_matrix(filename, metadata, center=True, max_snapshots=None):

    if max_snapshots==None:
        n_frames = metadata['n_frames']
    else:
        n_frames = min(max_snapshots, metadata['n_frames'])
    
    n_atoms = metadata['n_atoms']
    n_features = n_atoms * 3
    
    print(f"\nAcquisition de {n_frames} snapshots du JSON...")
    print(f"Creation d'une matrice de taille : ({n_frames} × {n_features}) = {(n_frames * n_features * 4) / (1024**3):.2f} GB")
    
    X = np.zeros((n_frames, n_features), dtype=np.float32)
    clusters = np.zeros(n_frames, dtype=np.int32)
    is_centroid = np.zeros(n_frames, dtype=bool)
    
    with open(filename, 'rb') as f:
        parser = ijson.items(f, 'snapshots.item')
        
        count = 0
        start_time = time.time()
        
        for snapshot in parser:
            if count >= n_frames:
                break
            
            snap_id = snapshot['id']
            cluster_id = snapshot['cluster']
            is_cent = snapshot['is_centroid']
            atoms = snapshot['atoms']
            
            for j, atom in enumerate(atoms):
                X[count, j*3 + 0] = atom['x']
                X[count, j*3 + 1] = atom['y']
                X[count, j*3 + 2] = atom['z']
            
            clusters[count] = cluster_id
            is_centroid[count] = is_cent
            
            count += 1
            
            if count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = count / elapsed
                eta = (n_frames - count) / rate if rate > 0 else 0
                print(f"  Loaded {count}/{n_frames} snapshots ({count/n_frames*100:.1f}%) - "
                      f"Rate: {rate:.1f} snap/s - ETA: {eta:.0f}s", end='\r')
        
        print()
    
    # Il faut centrer les snapshots
    if center:
        print("Centrage des snapshots...")
        for i in range(n_frames):
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{n_frames}...", end='\r')
            coords = X[i].reshape(n_atoms, 3)
            centroid = coords.mean(axis=0)
            coords -= centroid
            X[i] = coords.flatten()
        print()
    
    elapsed = time.time() - start_time
    print(f"Fin de streaming en {elapsed:.2f} secondes")
    
    return X, clusters, is_centroid


def apply_incremental_pca(X, n_components=10, batch_size=1000):

    print("Début du calcul PCA")
    print("Components: {n_components}")
    print("Batch size: {batch_size}")
    
    start_time = time.time()
    
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    

    n_samples = X.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"Fitting PCA in {n_batches} batches...")
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = X[i:end_idx]
        pca.partial_fit(batch)
        
        batch_num = i // batch_size + 1
        print(f"Batch {batch_num}/{n_batches} processed", end='\r')
    
    print()
    
    # Transform in batches
    print(f"Separation des donnees en {n_batches} batches...")
    X_pca = np.zeros((n_samples, n_components), dtype=np.float32)
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = X[i:end_idx]
        X_pca[i:end_idx] = pca.transform(batch)
        
        batch_num = i // batch_size + 1
        print(f"Batch {batch_num}/{n_batches} transforme", end='\r')
    
    print()
    
    elapsed = time.time() - start_time
    print(f"Calcul PCA fini en {elapsed:.2f} secondes")
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print("\nExplained variance:")
    for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
        print(f"PC{i+1}: {var*100:.2f}% (cumulative: {cum_var*100:.2f}%)")
    
    return X_pca, pca

def plot_pca_2d(X_pca, clusters, is_centroid, metadata,maxsnp=30003):
    
    print("\nCreating 2D plot...")
    
    n_clusters = metadata['n_clusters']
    colors = px.colors.qualitative.Plotly[:n_clusters]
    
    # Downsample for display if too many points
    n_points = X_pca.shape[0]
    max_display_points = 10000
    
    if n_points > max_display_points:
        print(f"  Downsampling from {n_points} to {max_display_points} points for display...")
        centroid_indices = np.where(is_centroid)[0]
         
        # Sample remaining points
        non_centroid_indices = np.where(~is_centroid)[0]
        n_to_sample = max_display_points - len(centroid_indices)
        
        if len(non_centroid_indices) > n_to_sample:
            sampled_indices = np.random.choice(non_centroid_indices, n_to_sample, replace=False)
        else:
            sampled_indices = non_centroid_indices
        
        display_indices = np.concatenate([centroid_indices, sampled_indices])
        
        X_display = X_pca[display_indices]
        clusters_display = clusters[display_indices]
        is_centroid_display = is_centroid[display_indices]
    else:
        display_indices = np.arange(n_points)
        X_display = X_pca
        clusters_display = clusters
        is_centroid_display = is_centroid
    
    fig = go.Figure()
    
    for k in range(n_clusters):
        # Regular points
        mask = (clusters_display == k) & (~is_centroid_display)
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=X_display[mask, 0],
                y=X_display[mask, 1],
                mode='markers',
                name=f'Cluster {k}',
                marker=dict(
                    size=4,
                    color=colors[k % len(colors)],
                    opacity=0.5,
                    line=dict(width=0)
                ),
                text=[f'Snapshot {display_indices[i]}' for i in np.where(mask)[0]],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))
        
        # # Centroids
        # centroid_mask = (clusters_display == k) & is_centroid_display
        # if centroid_mask.sum() > 0:
        #     fig.add_trace(go.Scatter(
        #         x=X_display[centroid_mask, 0],
        #         y=X_display[centroid_mask, 1],
        #         mode='markers',
        #         name=f'Centroid {k}',
        #         marker=dict(
        #             size=15,
        #             color=colors[k % len(colors)],
        #             symbol='diamond',
        #             line=dict(color='black', width=2)
        #         ),
        #         text=[f'CENTROID {display_indices[i]}' for i in np.where(centroid_mask)[0]],
        #         hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
        #     ))
    
    title = f'PCA Visualization (2D)<br><sub>{maxsnp}/{metadata["n_frames"]} snapshots'
    if n_points > max_display_points:
        title += f' (showing {max_display_points} sampled points)'
    title += '</sub>'
    
    fig.update_layout(
        title=title,
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        width=1200,
        height=800,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def plot_variance_explained(pca):
    """Plot cumulative variance explained by principal components"""
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(len(explained_var))],
        y=explained_var * 100,
        name='Individual',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(cumulative_var))],
        y=cumulative_var * 100,
        name='Cumulative',
        mode='lines+markers',
        marker=dict(size=8, color='red'),
        line=dict(width=3, color='red')
    ))
    
    fig.update_layout(
        title='Variance Explained by Principal Components',
        xaxis_title='Principal Component',
        yaxis_title='Variance Explained (%)',
        width=1000,
        height=600,
        showlegend=True
    )
    
    return fig

def main():
    try:
        import ijson
    except ImportError:
        print("ERROR: ijson not installed!")
        print("Install it with: pip install ijson")
        sys.exit(1)
    
    # Configuration
    FILENAME ='dataset/clustering_results.json'
    N_COMPONENTS = 10
    BATCH_SIZE = 1000
    MAX_SNAPSHOTS = 5000
    
    metadata = get_metadata(FILENAME)
    centroids = get_centroids(FILENAME)
    
    # Estimate memory usage
    n_frames = metadata['n_frames'] if MAX_SNAPSHOTS is None else min(MAX_SNAPSHOTS, metadata['n_frames'])
    n_atoms = metadata['n_atoms']
    estimated_gb = (n_frames * n_atoms * 3 * 4) / (1024**3)
    
    print("Estimate : {estimated_gb:.2f} GB")
    
    X, clusters, is_centroid = stream_snapshots_to_matrix(FILENAME, metadata, center=True,max_snapshots=MAX_SNAPSHOTS)
    
    X_pca, pca = apply_incremental_pca(X, n_components=N_COMPONENTS, batch_size=BATCH_SIZE)
    

    del X
    
    print("Rendu des visualisations...")
    
    # 2D plot
    fig_2d = plot_pca_2d(X_pca[:, :2], clusters, is_centroid, metadata,maxsnp=MAX_SNAPSHOTS)
    fig_2d.write_html('output/pca_2d.html')
    print("2D Plot : output/pca_2d.html")
    
    # Variance
    fig_var = plot_variance_explained(pca)
    fig_var.write_html('output/pca_variance.html')
    print("Variance plot : output/pca_variance.html")
    
    print("Done")


if __name__ == '__main__':
    main()