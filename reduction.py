# key reduction functions - formulas can be found in our paper: https://openreview.net/pdf?id=9GUTgHZgKCH
# these implementations are a bit better than the original ones on the ETH D-MATH Gitlab, but still not perfect
# later should implement a auto-reduction function that will automatically reduce the model given a certain criterion or tolerance

import numpy as np
import torch

# 1st step

def remove_outside_neurons(weights, inputs):
    """
    Remove neurons that are always on or always off

    Args:
    weights: Tuple[torch.Tensor]
        The weights of the model, format: (v, b, w, c, w_affine)
    inputs: torch.Tensor
        The local input tensor to the current stack, shape: (batch_size, input_dim)
    
    Returns:
    new_weights: Tuple[torch.Tensor]
        The new weights of the model, format: (v, b, w, c, w_affine)
    """
    if len(weights) == 5:
        v, b, w, c, w_affine = weights
    elif len(weights) == 4:
        v, b, w, c = weights
        # w_affine = torch.zeros((v.shape[0], c.shape[0]))

    bools = (v @ inputs.T) > 0
    out_neg_index = (bools.sum(1) == 0)
    out_pos_index = (bools.sum(1) == 100)
    out_index = out_neg_index | out_pos_index

    if len(weights) == 5:
        affine_weights = (
            w_affine
            + w.T[out_pos_index].T
            @ v[out_pos_index]
        )
    elif len(weights) == 4:
        affine_weights = (
            w.T[out_pos_index].T
            @ v[out_pos_index]
        )
    bias = (
        c + 
        w.T[out_pos_index].T
        @ b[out_pos_index]
    )
    new_weights = (v[~out_index], b[~out_index], w.T[~out_index].T, bias, affine_weights)
    return new_weights


# 2nd step


def remove_weak_neurons(weights, discard_neurons=100):
    """
    Remove neurons that are weakly connected to the input

    Args:
    weights: Tuple[torch.Tensor]
        The weights of the model, format: (v, b, w, c, w_affine)
    
    Returns:
    new_weights: Tuple[torch.Tensor]
        The new weights of the model, format: (v_new, b_new, w_new, c, w_affine)
    """
    v, b, w, c, w_affine = weights

    weights = torch.sqrt(v.norm(p=1, dim=1)**2 + b**2) * w.norm(p=1, dim=0)
    index = torch.argsort(weights)
    new_weights = [v[index][discard_neurons:], b[index][discard_neurons:], w[:, index][:, discard_neurons:], c, w_affine]
    return new_weights

# 3rd step


def compute_bvv(weights):
    """
    Compute the bvv matrix from the weights

    Args:
    weights: Tuple[torch.Tensor]
        The weights of the model, format: (v, b, w, c, w_affine)

    Returns:
    bvv: torch.Tensor
        The bvv matrix
    """
    v, b, *_ = weights
    # norms = torch.norm(v, p=1, dim=1)
    # bvv = (torch.stack([b for _ in range(v.shape[1])]).T * v / torch.stack([norms**2 for _ in range(v.shape[1])]).T)
    # v_normed = v / torch.stack([norms for _ in range(v.shape[1])]).T
    bvv = b.view(-1, 1) * v / v.norm(dim=1, keepdim=True)**2
    v_normed = v / v.norm(dim=1, keepdim=True)
    return torch.cat([bvv, v_normed], dim=1)


def compute_distance_matrix_for_clustering_merge_signs(weights, ord=1, verbose=1):
    # """from usual weights, compute bv/|v|**2 which will be later clustered
    # they will then be concatenated with v/|v|"""
    v, b, _, _, _, _ = weights
    norms = np.linalg.norm(v, axis=0, ord=ord)
    bvv = np.array([b[i] * v[:, i] / norms[i] ** 2 for i, _ in enumerate(v[0])])

    v_normed = v / norms
    # bvv = np.concatenate((bvv, v_normed.T), axis=1)

    cos_sim = lambda v, v_: 1 - np.abs(
        v @ v_.T / (np.linalg.norm(v) * np.linalg.norm(v_))
    )
    # cos_sim = lambda v, v_: 1 - v @ v_.T / (np.linalg.norm(v) * np.linalg.norm(v_))
    eucl_dist = lambda v, v_: np.linalg.norm(v - v_)

    D = np.zeros((norms.shape[0], norms.shape[0]))
    cluster_weights = compute_weights_for_clustering(weights, ord=ord, verbose=verbose)

    for i, m in enumerate(D):
        for j, _ in enumerate(m):
            if j > i:
                D[i, j] = eucl_dist(bvv[i, :], bvv[j, :]) + cos_sim(
                    v_normed[:, i], v_normed[:, j]
                )
                D[i, j] *= min(cluster_weights[i], cluster_weights[j])
            elif i > j:
                D[i, j] = D[j, i]
            else:
                D[i, j] = 0
    if verbose:
        print(f"Done computing M for clustering. Shape of bvv: {D.shape}")
    return D


def get_cluster_weights(weights):
    """
    Compute the weighting for the clustering

    Args:
    weights: Tuple[torch.Tensor]
        The weights of the model, format: (v, b, w, c, w_affine)

    Returns:
    Tuple[torch.Tensor]
        The weights of the cluster, format: (v.shape[1])
    """
    v, _, w, _, _ = weights
    v = v.norm(dim=1)
    w = w.norm(dim=0)
    return v * w


from sklearn.cluster import KMeans, AgglomerativeClustering

def cluster_bvv(clusters, bvv, cluster_weights=None, verbose=True):
    """
    Cluster bvv using KMeans

    Args:
    clusters : int
        Number of clusters
    bvv : torch.Tensor
        The BVV matrix
    cluster_weights : torch.Tensor
        Weights for clustering
    verbose : bool
        Print progress

    Returns:
    kmeans : KMeans
        The KMeans object
    """
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(
        bvv, sample_weight=cluster_weights
    )
    if verbose == 1:
        print(
            f"Done clustering bvv. Shape of centers: {kmeans.cluster_centers_.shape}"
        )
    return kmeans


def cluster_neurons_merge_signs(clusters, D, linkage="complete", verbose=True):
    """
    returns kmeans object that fitted C clusters to bvv
    make sure to import KMeans from sklearn.cluster before using
    weights are for clustering
    """
    # kmeans = KMeans(n_clusters=clusters, random_state=0).fit(
    #     bvv, sample_weight=cluster_weights
    # )
    agglo = AgglomerativeClustering(
        n_clusters=clusters,
        affinity="precomputed",
        linkage=linkage,
    ).fit(D)
    # if verbose == 1:
    #     print(
    #         f"Done clustering neurons. Shape of centers: {kmeans.cluster_centers_.shape}"
    #     )
    return agglo


### SELECTION
def g_function(xi):
    """
    g(xi) as specified by the formulas
    """
    return 1 / torch.sqrt(xi**2 + 1)


def get_kink(tmp_weights):
    """
    Compute the kink of the cluster

    Args:
    tmp_weights: Tuple[torch.Tensor]
        The weights of the model, format: (v, b, w)

    Returns:
    xi: torch.Tensor
        The kink of the cluster
    """
    v, b, w = tmp_weights
    xi = b @ v.norm(dim=1)
    xi /= w.norm(dim=1) @ v.norm(dim=1)
    return xi


def get_strength(tmp_weights):
    """
    Compute the strength of the cluster

    Args:
    tmp_weights: Tuple[torch.Tensor]
        The weights of the model, format: (v, b, w)

    Returns:
    strength: torch.Tensor
        The strength of the cluster
    """
    v, _, w = tmp_weights
    return torch.norm(w.T @ v.norm(dim=1))


def get_pos_index(weights, ord=1):
    v, _, w = weights
    strength_vector = lambda v, w: np.linalg.norm(
        w * np.linalg.norm(v, ord=1, axis=1, keepdims=True), axis=1, ord=1
    )
    s = strength_vector(v, w)
    ref = s.argmax()
    dot_products = np.array([v[ref] @ vee for vee in v])
    return dot_products > 0


def get_w(tmp_weights, s, g_xi):
    """
    Compute the w matrix of the cluster

    Args:
    tmp_weights: Tuple[torch.Tensor]
        The weights of the model, format: (v, b, w)
    s: torch.Tensor
        The s value
    g_xi: torch.Tensor
        The g(xi) value

    Returns:
    w: torch.Tensor
        The new weight w of the cluster
    """
    v, _, w = tmp_weights
    return (w.T @ v.norm(dim=1) / torch.sqrt(s * g_xi))


def get_v(tmp_weights, s, g_xi):
    """
    Compute the v matrix of the cluster

    Args:
    tmp_weights: Tuple[torch.Tensor]
        The weights of the model, format: (v, b, w)
    s: torch.Tensor
        The s value
    g_xi: torch.Tensor
        The g(xi) value

    Returns:
    v: torch.Tensor
        The new weight v of the cluster
    """
    v, _, w = tmp_weights
    new_v = w.norm(dim=1) @ v
    new_v /= (w.norm(dim=1) @ v).norm()
    return new_v * torch.sqrt(s * g_xi)


def new_v_affine(weights, s_neg, ord=1, pos_indices=None):
    v, _, w = weights
    din = v.shape[1]
    neg_indices = np.logical_not(pos_indices)
    norm_w = np.linalg.norm(w, ord=ord, axis=1, keepdims=True)
    v_new = np.sum(norm_w[neg_indices] * v[neg_indices], axis=0)
    v_new /= np.linalg.norm(v_new, ord=ord)
    v_new *= np.sqrt(s_neg)
    v_new = v_new.reshape((din, 1))
    return v_new


def new_w_affine(weights, s_neg, ord=1, pos_indices=None):
    v, _, w = weights
    dout = w.shape[1]
    neg_indices = np.logical_not(pos_indices)
    norm_v = np.linalg.norm(v, ord=ord, axis=1, keepdims=True)
    w_new = np.sum(norm_v[neg_indices] * w[neg_indices], axis=0)
    # w_new /= np.linalg.norm(w_new, ord=ord)
    w_new /= np.sqrt(s_neg)
    w_new = w_new.reshape((1, dout))
    return w_new


def get_b(v, xi):
    """
    Compute the new b value

    Args:
    v: torch.Tensor
        The v matrix
    xi: torch.Tensor
        The xi value

    Returns:
    b: torch.Tensor
        The new b value
    """
    return -xi * v.norm()


def get_clustered_weights(weights, cluster):
    """
    Get the weights of the model clustered

    Args:
    weights: Tuple[torch.Tensor]
        The weights of the model, format: (v, b, w, c, w_affine)
    cluster: KMeans
        The cluster object

    Returns:
    new_weights: List[Tuple[torch.Tensor]]
        The new weights of the model, format: [(v_new, b_new, w_new, c, w_affine)]
    """
    v, b, w, *_ = weights
    v = v.detach()
    b = b.detach()
    w = w.T.detach()
    V, B, W = [], [], []
    for i in range(cluster.cluster_centers_.shape[0]):
        v_temp, b_temp, w_temp = (
            v[cluster.labels_ == i],
            b[cluster.labels_ == i],
            w[cluster.labels_ == i],
        )
        temp_weights = [v_temp, b_temp, w_temp]
        xi = get_kink(temp_weights)
        g_xi = g_function(xi)
        s = get_strength(temp_weights)
        w_new = get_w(temp_weights, s, g_xi)
        W.append(w_new)
        v_new = get_v(temp_weights, s, g_xi)
        V.append(v_new)
        b_new = get_b(v_new, xi)
        B.append(b_new)
    return [torch.stack(V), torch.stack(B), torch.stack(W).T, weights[3], weights[4]]


def cluster_neurons(weights, n_clusters=10, cluster_weights=None):
    """
    Cluster the neurons of the model

    Args:
    weights: Tuple[torch.Tensor]
        The weights of the model, format: (v, b, w, c, w_affine)
    n_clusters: int
        The number of clusters

    Returns:
    new_weights: List[Tuple[torch.Tensor]]
        The new weights of the model, format: [(v_new, b_new, w_new, c, w_affine)]
    """
    bvv = compute_bvv(weights)
    cluster = cluster_bvv(n_clusters, bvv.detach(), cluster_weights=cluster_weights)
    return get_clustered_weights(weights, cluster)


# 4th step


def svd_affine(weights, affine_neurons_left=1):
    """
    Set the weights of the model with the SVD of the last affine transformation

    Args:
    weights: Tuple[torch.Tensor]
        The weights of the model, format: (v, b, w, c, w_affine)
    affine_neurons_left: int
        The number of neurons to keep in the last affine transformation

    Returns:
    new_weights: List[Tuple[torch.Tensor]]
        The new weights of the model, format: [(v_new, b_new, w_new, c, P, Q)]
    """
    v, b, w, c, w_affine = weights
    U, sigma, V = torch.svd(w_affine)
    # Just a note for me:
    # U @ sigma @ V.T - w_affine              # timeit says 18.2 µs
    # U @ torch.diag(sigma) @ V.T - w_affine  # timeit says 34.5 µs
    sigma_sqrt = torch.diag(torch.sqrt(sigma[:affine_neurons_left]))
    P = sigma_sqrt @ V.T[:affine_neurons_left]
    Q = U[:,:affine_neurons_left] @ sigma_sqrt
    return [v, b, w, c, P, Q]