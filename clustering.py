"""Spectral Clustering Implementation"""
# This code is adapted from https://github.com/dgerosa/gwlabel (Gerosa, 2024)

import time
import torch
import numpy as np
np.random.seed(0)

import scipy.sparse as sp

from tensordict import TensorDict
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import sort_graph_by_row_values

def kneighbors_graph_torch(X, k=10):
    """
    Compute the k-nearest neighbors connectivity graph as a sparse matrix using PyTorch.
    This mimics sklearn.neighbors.kneighbors_graph(..., mode="connectivity").
    
    Parameters:
        X (torch.Tensor): Input tensor of shape (n_samples, n_features)
        k (int): Number of neighbors (excluding self)
    
    Returns:
        scipy.sparse.csr_matrix: Sorted, symmetric sparse connectivity (binary) k-NN graph
    """
    device = X.device
    n_samples = X.shape[0]
    k = n_samples if k is None else k

    # Compute Euclidean (L2) distances
    distances = torch.cdist(X, X, p=1)

    # Exclude self from neighbors by setting the diagonal to infinity
    distances[torch.arange(n_samples, device=device), torch.arange(n_samples, device=device)] = float('inf')
    
    # Get the indices of the k nearest neighbors (excluding self)
    _, knn_indices = torch.topk(distances, k=k, largest=False, sorted=True)

    # Build row indices (each row repeated k times)
    row_indices = torch.arange(n_samples, device=device).repeat_interleave(k)
    col_indices = knn_indices.flatten()
    
    # For connectivity mode, set all nonzero values to 1
    data = torch.ones_like(col_indices, dtype=torch.float32, device=device)

    # Optionally sort the rows (sklearn requires a sorted CSR matrix)
    sorted_order = torch.argsort(row_indices)
    row_indices = row_indices[sorted_order]
    col_indices = col_indices[sorted_order]
    data = data[sorted_order]

    # Create a PyTorch sparse tensor
    knn_graph = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]), 
        values=data, 
        size=(n_samples, n_samples)
    )
    knn_graph = knn_graph.coalesce()

    # Convert to SciPy CSR sparse matrix
    indices = knn_graph.indices().cpu().numpy()
    values = knn_graph.values().cpu().numpy()
    knn_matrix = sp.csr_matrix((values, (indices[0], indices[1])), shape=(n_samples, n_samples))
    
    # Sort rows (avoids warnings in sklearn)
    knn_matrix = sort_graph_by_row_values(knn_matrix, warn_when_not_sorted=False)
    
    # Symmetrize the graph (sklearn does this internally for nearest_neighbors)
    knn_matrix = knn_matrix.maximum(knn_matrix.transpose())
    
    return knn_matrix



class PosteriorClustering:
    """
        Clustering of posterior samples using spectral clustering.

        Args:
        -----
            n_clusters (int): Number of clusters to fit.
            cluster_on (str): The posterior param (flag) to cluster on. Default is ['tcoal']. Can be a list of variables.
            k          (int): Number of neighbors for the k-NN graph. Default is 5000.
    """
    def __init__(self, n_clusters=3, cluster_on=['tcoal'], k=5000, verbose=False):

        self.n_clusters = n_clusters
        self.cluster_on = cluster_on
        self.verbose    = verbose
        self.k = k
    
    
    @property
    def clustering(self):
        """We define as a property so that every time we call it, it returns a new instance."""
        return SpectralClustering(n_clusters    = self.n_clusters,
                                  affinity      = 'precomputed_nearest_neighbors',
                                  assign_labels = 'cluster_qr',
                                  eigen_solver  = "amg",
                                  n_jobs        = -1,
                                  random_state  = 0)
    
    @staticmethod
    def group_by(keys, n):
        return [keys[i:i+n] for i in range(0, len(keys), n)]
    

    @staticmethod
    def posterior_group_by(posterior, grouped_params):
        """
            Group the posterior samples by the given parameters.

            Args:
            -----
                posterior (TensorDict): The posterior samples.
                grouped_params (list): The list of grouped parameters.

            Returns:
            --------
                list  : The grouped posterior samples.
                tensor: The grouped posterior samples as a tensor.
        """
        grouped_list = []
        for group in grouped_params:
            g_ = torch.stack([posterior[param] for param in group], dim=-1)
            grouped_list.append(g_)
        
        #concatenate the list of tensors to get a single tensor of shape [Nsamples, Nparams_to_cluster_on]
        grouped = torch.cat(grouped_list)
        return grouped_list, grouped


    def preprocess(self, posterior):
        """
            Preprocess the posterior data

            Args:
            -----
                posterior (TensorDict): The posterior samples.
        """
        # # Select the posterior params to cluster on
        cluster_params = [param for param in posterior.keys() if any([param.startswith(flag) for flag in self.cluster_on])]
        grouped_cluster_params = self.group_by(cluster_params, len(self.cluster_on))
        
        if self.verbose: print(grouped_cluster_params)

        Xsplit, X = self.posterior_group_by(posterior, grouped_cluster_params)
        
        if self.verbose: print(f"Posterior shape: {X.shape}")
        
        # Normalize the data
        mean = X.mean(0, keepdim=True)
        std  = X.std(0,  keepdim=True)
        Xnorm = (X - mean) / std

        return Xsplit, Xnorm

    def reorder_labels(self, Xsplit, newlabels):
        """
        Reorder the labels to match the true labels using one-to-one matching via the Hungarian algorithm.
        
        Parameters:
            Xsplit (list of torch.Tensor): List of tensors for each true group, each of shape [Nsamples, n_params]
            newlabels (np.ndarray): Array of cluster labels from spectral clustering.
        
        Returns:
            np.ndarray: The reordered labels.
        """
        # Compute medians for the original groups.
        # Xsplit is assumed to be a list of tensors, each corresponding to one true group.
        original_medians = []
        for group in Xsplit:
            # Compute median along the sample dimension (dim=0) for each group.
            median = group.median(dim=0).values.cpu().numpy()  
            original_medians.append(median)
        
        # Compute medians for the clusters.
        unique_labels = np.unique(newlabels)
        cluster_medians = []
        # Concatenate Xsplit to have all samples together.
        X = torch.cat(Xsplit, dim=0)  # shape: [total_samples, n_params]
        for label in unique_labels:
            cluster_points = X[newlabels == label]
            median = cluster_points.median(dim=0).values.cpu().numpy()
            cluster_medians.append(median)
        
        # Build a cost matrix where cost[i, j] is the mean absolute difference 
        # between the median of cluster i and the median of original group j.
        n_clusters  = len(cluster_medians)
        n_groups    = len(original_medians)
        cost_matrix = np.zeros((n_clusters, n_groups))
        for i in range(n_clusters):
            for j in range(n_groups):
                cost_matrix[i, j] = np.mean(np.abs(cluster_medians[i] - original_medians[j]))
        
        # Use the Hungarian algorithm to solve the assignment problem.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Build the mapping from cluster label (from spectral clustering) to the desired label.
        # row_ind[i] corresponds to unique_labels[row_ind[i]] being matched to original group col_ind[i].
        mapping = { unique_labels[row]: col for row, col in zip(row_ind, col_ind) }
        
        # Reorder newlabels according to the mapping.
        reordered_labels = np.array([mapping[label] for label in newlabels])
        
        return reordered_labels
    
    def reorder_posterior_samples(self, posterior, newlabels):
        """
            Reorder the posterior samples to match the new labels.

            Args:
            -----
                posterior (TensorDict): The posterior samples.
                newlabels (np.ndarray): The new labels.

            Returns:
            --------
                TensorDict: The reordered posterior samples.
        """
        params = list(posterior.keys())
        _, X = self.posterior_group_by(posterior, self.group_by(params, len(params)//self.n_clusters))
        
        Xcluster = [X[newlabels == i] for i in np.unique(newlabels)]
        min_len = min([len(Xcluster[i]) for i in range(self.n_clusters)])

        Xcluster = torch.cat([Xcluster[i][:min_len] for i in range(len(Xcluster))], 1)

        reordered_posterior = dict()     
        for i, key in enumerate(posterior.keys()):
            reordered_posterior[key] = Xcluster[:, i]   
        
        return TensorDict.from_dict(reordered_posterior)

    
    def run_clustering(self, posterior):
        """
            Run the clustering on the posterior samples.

            Args:
            -----
                posterior (TensorDict): The posterior samples.
        """
        # Preprocess the data
        N = len(posterior[list(posterior.keys())[0]])
        Xsplit, Xnorm = self.preprocess(posterior)        
        
        # Compute the binary (connectivity) k-NN graph
        start = time.time()
        knn_connectivity = kneighbors_graph_torch(Xnorm, self.k)
        if self.verbose: print(f"Connectivity graph computation took: {time.time() - start:.3f} s")

        # Perform spectral clustering using the precomputed nearest-neighbors graph
        cluster_fit = self.clustering.fit(knn_connectivity)

        # Get the new labels & reorder them
        newlabels = self.reorder_labels(Xsplit, cluster_fit.labels_)
        
        #reorder posterior
        if self.verbose: print('Reordering posterior samples...')
        newposterior = self.reorder_posterior_samples(posterior, newlabels)
        
        return newlabels, newposterior
    
    def __call__(self, posterior):
        return self.run_clustering(posterior)


#----------------------
# Test the clustering
#----------------------
converter = {'M'                  : 'M',
             'q'                  : 'q',
             'Mchirp'             : '\mathcal{M}',
             'tcoal'              : 't'}


def get_latex_label(name):
    par_name, kind, num = name.split('_')
    return rf'${converter[par_name]}_{{{num}}}$'

if __name__=="__main__":
    import shutil
    import pandas as pd
    from tensordict import TensorDict
    from corner import corner
    import matplotlib.pyplot as plt

    # Load the posterior samples
    N = 10000
    name = 'posterior_samples.csv'
    posterior_file = f'training_results/KENN_3_BBH_16s_BEST/{name}'
    posterior = pd.read_csv(posterior_file)[:N]

    #convert to tensordict
    posterior_dict = {p: torch.from_numpy(posterior[p].values).float() for p in posterior.columns}
    posterior = TensorDict(posterior_dict)#.to('cuda:1')


    #run the clustering
    cluster = PosteriorClustering(k=2000, verbose=True, cluster_on=["tcoal"])

    start=time.time()
    labels, newposterior = cluster(posterior)
    print(f"Clustering took: {time.time() - start:.3f} s")
    

    old_posterior = torch.stack([posterior[param] for param in list(posterior.keys())], dim=-1).cpu().numpy()
    
    new_posterior = torch.stack([newposterior[param] for param in list(newposterior.keys())], dim=-1).cpu().numpy()
    all_keys = list(posterior.keys())

    truth = np.array([882.178,
                  274.961 ,
                  0.210,
                  -7.019 ,
                  802.606  ,
                  304.549 ,
                  0.377 ,
                  -7.621 ,
                  711.220,
                  273.514 ,
                  0.397  ,
                  -12.749  ])
    
    latex_labels = [get_latex_label(key) for key in all_keys]
    

    fsize = 22
    plt.rcParams["font.size"] = fsize

    if shutil.which("latex"):
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "Computer Modern"
        

    fig = None
    fig = corner(old_posterior, color="darkblue", bins = 30, labels=latex_labels, 
                quantiles=[0.16, 0.5, 0.84], show_titles=False, title_fmt='.2f', title_kwargs={"fontsize": 12},
                label_kwargs={"fontsize": 12}, hist_kwargs={"density": True, "histtype":"stepfilled", "alpha":0.1, "linewidth":2}, smooth=2.0, 
                plot_datapoints=False, plot_density=True, plot_contours=True, no_fill_contours=False,)

    fog = corner(old_posterior, color="darkblue", bins = 30, labels=latex_labels, 
                quantiles=[0.16, 0.5, 0.84], show_titles=False, title_fmt='.2f', title_kwargs={"fontsize": 12},
                label_kwargs={"fontsize": 12}, hist_kwargs={"density": True}, smooth=2.0, 
                plot_datapoints=False, plot_density=True, plot_contours=True, no_fill_contours=False, fig=fig)


    fig = corner(new_posterior, labels=latex_labels, bins=30, quantiles=[0.16, 0.5, 0.84], show_titles=False, title_fmt='.2f', title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 12}, hist_kwargs={"density": True, "histtype":"stepfilled", "alpha":0.15}, color='darkred', 
                fig=fig, truth_color='darkorange', smooth=2.0, plot_datapoints=False, plot_density=True, plot_contours=True, no_fill_contours=False, truth_kwargs={"linewidth": 2})


    fig = corner(new_posterior, labels=latex_labels, bins=30, quantiles=[0.16, 0.5, 0.84], show_titles=False, title_fmt='.2f', title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 25}, hist_kwargs={"density": True}, color='darkred', 
        truths=truth, fig=fig, truth_color='darkorange', smooth=2.0, plot_datapoints=False, plot_density=True, plot_contours=True, no_fill_contours=False, truth_kwargs={"linewidth": 2})

    # add legend
    fig.legend(
            handles  = [plt.Line2D([0], [0], color=color, lw=4) for color in ["darkblue", "darkred"]],
            labels   = ["Original posterior", "After Clustering"],
            loc      = "upper right",
            fontsize = 35, 
            frameon  = False
        )

    #plt.savefig("all_param_clustering.pdf", dpi=300, bbox_inches="tight")

    #save the fig object as a pdf
    fig.savefig("clustering.pdf", dpi=200, bbox_inches="tight")
