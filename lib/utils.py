import os
import re
import torch 
import logging 
import math
from collections import defaultdict

import open3d as o3d
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def dict_all_to_device(tensor_dict, device):
    """
    Puts all the tensors to a specified device

    Args: 
        tensor_dict (dict): dictionary of all tensors
        device (str): device to be used (cuda or cpu)

    """

    for key in tensor_dict:
        if isinstance(tensor_dict[key], torch.Tensor):
            if 'sinput' not in key:
                tensor_dict[key] = tensor_dict[key].to(device)
            

def save_checkpoint(filename, epoch, it, model, optimizer=None, scheduler=None, config=None, best_val=None):
    """
    Saves the current model, optimizer, scheduler, and side information to a checkpoint

    Args: 
        filename (str): path to where the checpoint will be saved
        epoch (int): current epoch
        it (int): current iteration
        model (nn.Module): torch neural network model
        optimizer (torch.optim): selected optimizer
        scheduler (torch.optim): selected scheduler
        config (dict): config parameters
        best_val (float): best validation result

    """
    
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'total_it': it,
        'optimizer': optimizer.state_dict(),
        'config': config,
        'scheduler': scheduler.state_dict(),
        'best_val': best_val,
    }

    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)


def load_checkpoint(model, optimizer, scheduler, filename):
    """
    Loads the saved checkpoint and updates the model, optimizer and scheduler.

    Args: 
        model (nn.Module): torch neural network model
        optimizer (torch.optim): selected optimizer
        scheduler (torch.optim): selected scheduler
        filename (str): path to the saved checkpoint

    Returns:
        model (nn.Module): model with pretrained parameters
        optimizer  (torch.optim): optimizer loaded from the checkpoint
        scheduler  (torch.optim): scheduler loaded from the checkpoint
        start_epoch (int): current epoch
        total_it (int): total number of iterations that were performed
        metric_val_best (float): current best valuation metric

    """
    start_epoch = 0
    total_it = 0
    metric_val_best = np.inf

    if os.path.isfile(filename):
        logging.info("Loading checkpoint {}".format(filename))
        checkpoint = torch.load(filename)

        # Safe loading of the model, load only the keys that are in the init and the saved model
        model_dict = model.state_dict()
        for key in model_dict:
            if key in checkpoint['state_dict']:
                model_dict[key] = checkpoint['state_dict'][key]

        model.load_state_dict(model_dict)

        if optimizer is not None and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                logging.info('could not load optimizer from the pretrained model')

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']

        if 'total_it' in checkpoint:
            total_it = checkpoint['total_it']

        if 'best_val' in checkpoint:
            metric_val_best = checkpoint['best_val']

        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
    else:
        logging.info("No checkpoint found at {}".format(filename))

    return model, optimizer, scheduler, start_epoch, total_it, metric_val_best


def load_point_cloud(file, data_type='numpy'):
    """
    Loads the point cloud coordinates from the '*.ply' file.
    Args: 
        file (str): path to the '*.ply' file
        data_type (str): data type to be returned (default: numpy)
    Returns:
        pc (np.array or open3d.PointCloud()): point coordinates [n, 3]
    """
    temp_pc = o3d.io.read_point_cloud(file)
     
    assert data_type in ['numpy', 'open3d'], 'Wrong data type selected when loading the ply file.' 
    
    if data_type == 'numpy':
        return np.asarray(temp_pc.points)
    else:         
        return temp_pc


def sorted_alphanum(file_list_ordered):
    """
    Sorts the list alphanumerically
    Args:
        file_list_ordered (list): list of files to be sorted
    Return:
        sorted_list (list): input list sorted alphanumerically
    """
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    sorted_list = sorted(file_list_ordered, key=alphanum_key)

    return sorted_list
    
def get_file_list(path, extension=None):
    """
    Build a list of all the files in the provided path
    Args:
        path (str): path to the directory 
        extension (str): only return files with this extension
    Return:
        file_list (list): list of all the files (with the provided extension) sorted alphanumerically
    """
    if extension is None:
        file_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        file_list = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)

    return file_list


def get_folder_list(path):
    """
    Build a list of all the files in the provided path
    Args:
        path (str): path to the directory 
        extension (str): only return files with this extension
    Returns:
        file_list (list): list of all the files (with the provided extension) sorted alphanumerically
    """
    folder_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    folder_list = sorted_alphanum(folder_list)
    
    return folder_list

def n_model_parameters(model):
    """
    Counts the number of parameters in a torch model
    Args:
        model (torch.Model): input model 
    
    Returns:
        _ (int): number of the parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pairwise_distance(src, dst, normalized=True):
    """Calculates squared Euclidean distance between each two points.
    Args:
        src (torch tensor): source data, [b, n, c]
        dst (torch tensor): target data, [b, m, c]
        normalized (bool): distance computation can be more efficient 
    Returns:
        dist (torch tensor): per-point square distance, [b, n, m]
    """

    if len(src.shape) == 2:
        src = src.unsqueeze(0)
        dst = dst.unsqueeze(0)

    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    # Minus such that smaller value still means closer 
    dist = -torch.matmul(src, dst.permute(0, 2, 1))

    # If inputs are normalized just add 1 otherwise compute the norms 
    if not normalized:
        dist *= 2 
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    
    else:
        dist += 1.0
    
    # Distances can get negative due to numerical precision
    dist = torch.clamp(dist, min=0.0, max=None)
    
    return dist


def rotation_error(R1, R2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})
    Args: 
        R1 (torch tensor): Estimated rotation matrices [b,3,3]
        R2 (torch tensor): Ground truth rotation matrices [b,3,3]
    Returns:
        ae (torch tensor): Rotation error in angular degreees [b,1]
    """
    R_ = torch.matmul(R1.transpose(1,2), R2)
    e = torch.stack([(torch.trace(R_[_, :, :]) - 1) / 2 for _ in range(R_.shape[0])], dim=0).unsqueeze(1)

    # Clamp the errors to the valid range (otherwise torch.acos() is nan)
    e = torch.clamp(e, -1, 1, out=None)

    ae = torch.acos(e)
    pi = torch.Tensor([math.pi])
    ae = 180. * ae / pi.to(ae.device).type(ae.dtype)

    return ae


def translation_error(t1, t2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})
    Args: 
        t1 (torch tensor): Estimated translation vectors [b,3,1]
        t2 (torch tensor): Ground truth translation vectors [b,3,1]
    Returns:
        te (torch tensor): translation error in meters [b,1]
    """
    return torch.norm(t1-t2, dim=(1, 2))


def kabsch_transformation_estimation(x1, x2, weights=None, normalize_w = True, eps = 1e-7, best_k = 0, w_threshold = 0, compute_residuals = False):
    """
    Torch differentiable implementation of the weighted Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm). Based on the correspondences and weights calculates
    the optimal rotation matrix in the sense of the Frobenius norm (RMSD), based on the estimated rotation matrix it then estimates the translation vector hence solving
    the Procrustes problem. This implementation supports batch inputs.
    Args:
        x1            (torch array): points of the first point cloud [b,n,3]
        x2            (torch array): correspondences for the PC1 established in the feature space [b,n,3]
        weights       (torch array): weights denoting if the coorespondence is an inlier (~1) or an outlier (~0) [b,n]
        normalize_w   (bool)       : flag for normalizing the weights to sum to 1
        best_k        (int)        : number of correspondences with highest weights to be used (if 0 all are used)
        w_threshold   (float)      : only use weights higher than this w_threshold (if 0 all are used)
    Returns:
        rot_matrices  (torch array): estimated rotation matrices [b,3,3]
        trans_vectors (torch array): estimated translation vectors [b,3,1]
        res           (torch array): pointwise residuals (Eucledean distance) [b,n]
        valid_gradient (bool): Flag denoting if the SVD computation converged (gradient is valid)
    """
    if weights is None:
        weights = torch.ones(x1.shape[0],x1.shape[1]).type_as(x1).to(x1.device)

    if normalize_w:
        sum_weights = torch.sum(weights,dim=1,keepdim=True) + eps
        weights = (weights/sum_weights)

    weights = weights.unsqueeze(2)

    if best_k > 0:
        indices = np.argpartition(weights.cpu().numpy(), -best_k, axis=1)[0,-best_k:,0]
        weights = weights[:,indices,:]
        x1 = x1[:,indices,:]
        x2 = x2[:,indices,:]

    if w_threshold > 0:
        weights[weights < w_threshold] = 0


    x1_mean = torch.matmul(weights.transpose(1,2), x1) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)
    x2_mean = torch.matmul(weights.transpose(1,2), x2) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)

    x1_centered = x1 - x1_mean
    x2_centered = x2 - x2_mean

    cov_mat = torch.matmul(x1_centered.transpose(1, 2),
                            (x2_centered * weights))

    try:
        u, s, v = torch.svd(cov_mat)

    except Exception as e:
        r = torch.eye(3,device=x1.device)
        r = r.repeat(x1_mean.shape[0],1,1)
        t = torch.zeros((x1_mean.shape[0],3,1), device=x1.device)

        res = transformation_residuals(x1, x2, r, t)

        return r, t, res, True

    tm_determinant = torch.det(torch.matmul(v.transpose(1, 2), u.transpose(1, 2)))

    determinant_matrix = torch.diag_embed(torch.cat((torch.ones((tm_determinant.shape[0],2),device=x1.device), tm_determinant.unsqueeze(1)), 1))

    rotation_matrix = torch.matmul(v,torch.matmul(determinant_matrix,u.transpose(1,2)))

    # translation vector
    translation_matrix = x2_mean.transpose(1,2) - torch.matmul(rotation_matrix,x1_mean.transpose(1,2))

    # Residuals
    res = None
    if compute_residuals:
        res = transformation_residuals(x1, x2, rotation_matrix, translation_matrix)

    return rotation_matrix, translation_matrix, res, False


def transformation_residuals(x1, x2, R, t):
    """
    Computer the pointwise residuals based on the estimated transformation paramaters
    
    Args:
        x1  (torch array): points of the first point cloud [b,n,3]
        x2  (torch array): points of the second point cloud [b,n,3]
        R   (torch array): estimated rotation matrice [b,3,3]
        t   (torch array): estimated translation vectors [b,3,1]
    Returns:
        res (torch array): pointwise residuals (Eucledean distance) [b,n,1]
    """
    x2_reconstruct = torch.matmul(R, x1.transpose(1, 2)) + t 

    res = torch.norm(x2_reconstruct.transpose(1, 2) - x2, dim=2)

    return res

def transform_point_cloud(x1, R, t):
    """
    Transforms the point cloud using the giver transformation paramaters
    
    Args:
        x1  (np array): points of the point cloud [b,n,3]
        R   (np array): estimated rotation matrice [b,3,3]
        t   (np array): estimated translation vectors [b,3,1]
    Returns:
        x1_t (np array): points of the transformed point clouds [b,n,3]
    """
    if len(R.shape) != 3:
        R = R.unsqueeze(0)

    if len(t.shape) != 3:
        t = t.unsqueeze(0)
    
    if len(x1.shape) != 3:
        x1 = x1.unsqueeze(0)

    x1_t = (torch.matmul(R, x1.transpose(2,1)) + t).transpose(2,1)

    return x1_t


def refine_ego_motion(pc_s, pc_t, bckg_mask_s, bckg_mask_t, R_est, t_est):
    """
    Refines the coarse ego motion estimate based on all background indices
    
    Args:
        pc_s  (torch.tensor): points of the source point cloud [b,n,3]
        pc_t  (torch.tensor): points of the target point cloud [b,n,3]
        bckg_mask_s  (torch.tensor): background mask for the source points [b,n]
        bckg_mask_t  (torch.tensor): background mask for the target points [b,n]
        R_est   (torch.tensor): coarse rotation matrices [b,3,3]
        t_est   (torch.tensor): coarse translation vectors [b,3,1]
    Returns:
        R_ref  (np array): refined transformation parameters [b,3,3]
        t_ref  (np array): refined transformation parameters [b,3,1]
    """

    pcd_s = o3d.geometry.PointCloud()
    pcd_t = o3d.geometry.PointCloud()

    R_est = R_est.cpu().numpy()
    t_est = t_est.cpu().numpy()

    R_ref = np.zeros_like(R_est)
    t_ref = np.zeros_like(t_est)

    init_T = np.eye(4)

    for b_idx in range(pc_s.shape[0]):
        xyz_bckg_s = pc_s[b_idx, bckg_mask_s[b_idx,:], :].cpu().numpy()
        xyz_bckg_t = pc_t[b_idx, bckg_mask_t[b_idx,:], :].cpu().numpy()

        pcd_s.points = o3d.utility.Vector3dVector(xyz_bckg_s)
        pcd_t.points = o3d.utility.Vector3dVector(xyz_bckg_t)

        init_T[0:3,0:3] = R_est[b_idx,:,:]
        init_T[0:3,3:4] = t_est[b_idx,:,:]

        trans = o3d.registration.registration_icp(pcd_s, pcd_t,
                                                  max_correspondence_distance=0.15, init=init_T,
                                                  criteria=o3d.registration.ICPConvergenceCriteria(max_iteration = 300))

        R_ref[b_idx,:,:] = trans.transformation[0:3,0:3]
        t_ref[b_idx,:,:] = trans.transformation[0:3,3:4]
    
    return R_ref, t_ref




def refine_cluster_motion(pc_s, pc_t, R_est=None, t_est=None):
    """
    Refines the motion of a foreground rigid agent (clust) 
    
    Args:
        pc_s  (torch.tensor): points of the cluster points [n,3]
        pc_t  (torch.tensor): foreground point of the target point cloud [m,3]
        R_coarse   (torch.tensor): coarse rotation matrices [3,3]
        t_coarse   (torch.tensor): coarse translation vectors [3,1]
    Returns:
        R_ref  (np array): refined transformation parameters [3,3]
        t_ref  (np array): refined transformation parameters [3,1]
    """

    pcd_s = o3d.geometry.PointCloud()
    pcd_t = o3d.geometry.PointCloud()

    init_T = np.eye(4, dtype=np.float)

    if R_est is not None:
        init_T[0:3,0:3] = R_est.cpu().numpy()
        init_T[0:3,3:4] = t_est.cpu().numpy()

    pcd_s.points = o3d.utility.Vector3dVector(pc_s.cpu())
    pcd_t.points = o3d.utility.Vector3dVector(pc_t.cpu())

    trans = o3d.registration.registration_icp(pcd_s, pcd_t,
                                              max_correspondence_distance=0.25, init=init_T,
                                              criteria=o3d.registration.ICPConvergenceCriteria(max_iteration = 300))

    R_ref = trans.transformation[0:3,0:3].astype(np.float32)
    t_ref = trans.transformation[0:3,3:4].astype(np.float32)
    
    return R_ref, t_ref


def compute_epe(est_flow, gt_flow, sem_label=None, eval_stats =False, mask=None):
    """
    Compute 3d end-point-error

    Args:
        st_flow (torch.Tensor): estimated flow vectors [n,3]
        gt_flow  (torch.Tensor): ground truth flow vectors [n,3]
        eval_stats (bool): compute the evaluation stats as defined in FlowNet3D
        mask (torch.Tensor): boolean mask used for filtering the epe [n]

    Returns:
        epe (float): mean EPE for current batch
        epe_bckg (float): mean EPE for the background points
        epe_forg (float): mean EPE for the foreground points
        acc3d_strict (float): inlier ratio according to strict thresh (error smaller than 5cm or 5%)
        acc3d_relax (float): inlier ratio according to relaxed thresh (error smaller than 10cm or 10%)
        outlier (float): ratio of outliers (error larger than 30cm or 10%)
    """

    metrics = {}
    error = est_flow - gt_flow
    
    # If mask if provided mask out the flow
    if mask is not None:
        error = error[mask > 0.5]
        gt_flow = gt_flow[mask > 0.5, :]
    
    epe_per_point = torch.sqrt(torch.sum(torch.pow(error, 2.0), -1))
    epe = epe_per_point.mean()

    metrics['epe'] = epe.item()


    if sem_label is not None:
        # Extract epe for background and foreground separately (background = class 0)
        bckg_mask = (sem_label == 0)
        forg_mask = (sem_label == 1)

        bckg_epe = epe_per_point[bckg_mask]
        forg_epe = epe_per_point[forg_mask]

        metrics['bckg_epe'] = bckg_epe.mean().item()
        metrics['bckg_epe_median'] = bckg_epe.median().item()
        
        if torch.sum(forg_mask) > 0:
            metrics['forg_epe_median'] = forg_epe.median().item()
            metrics['forg_epe'] = forg_epe.mean().item()

    if eval_stats:
        
        gt_f_magnitude = torch.norm(gt_flow, dim=-1)
        gt_f_magnitude_np = np.linalg.norm(gt_flow.cpu(), axis=-1)
        relative_err = epe_per_point / (gt_f_magnitude + 1e-4)
        acc3d_strict = (
            (torch.logical_or(epe_per_point < 0.05, relative_err < 0.05)).type(torch.float).mean()
        )
        acc3d_relax = (
            (torch.logical_or(epe_per_point < 0.1, relative_err < 0.1)).type(torch.float).mean()
        )
        outlier = (torch.logical_or(epe_per_point > 0.3, relative_err > 0.1)).type(torch.float).mean()

        metrics['acc3d_s'] = acc3d_strict.item()
        metrics['acc3d_r'] = acc3d_relax.item()
        metrics['outlier'] = outlier.item()

    return metrics


def compute_l1_loss(est_flow, gt_flow):
    """
    Compute training loss.

    Args:
    est_flow (torch.Tensor): estimated flow
    gt_flow (torch.Tensor): : ground truth flow

    Returns
    loss (torch.tensor): mean l1 loss of the current batch

    """

    error = est_flow - gt_flow
    loss = torch.mean(torch.abs(error))

    return loss



def precision_at_one(pred, target):
    """
    Computes the precision and recall of the binary fg/bg segmentation

    Args:
    pred (torch.Tensor): predicted foreground labels
    target (torch.Tensor): : gt foreground labels

    Returns
    precision_f (float): foreground precision
    precision_b (float): background precision
    recall_f (float): foreground recall
    recall_b (float): background recall

    """

    precision_f = (pred[target == 1] == 1).float().sum() / ((pred == 1).float().sum() + 1e-6)
    precision_b = (pred[target == 0] == 0).float().sum() / ((pred == 0).float().sum() + 1e-6)

    recall_f = (pred[target == 1] == 1).float().sum() / ((target == 1).float().sum() + 1e-6)
    recall_b = (pred[target == 0] == 0).float().sum() / ((target == 0).float().sum() + 1e-6)

    return precision_f, precision_b, recall_f, recall_b


def evaluate_binary_class(pred, target):
    """
    Computes the number of true/false positives and negatives

    Args:
    pred (torch.Tensor): predicted foreground labels
    target (torch.Tensor): : gt foreground labels

    Returns
    true_p (float): number of true positives
    true_n (float): number of true negatives
    false_p (float): number of false positives
    false_n (float): number of false negatives

    """

    true_p = (pred[target == 1] == 1).float().sum()
    true_n = (pred[target == 0] == 0).float().sum()

    false_p = (pred[target == 0] == 1).float().sum()
    false_n = (pred[target == 1] == 0).float().sum()

    return true_p, true_n, false_p, false_n



def upsample_flow(xyz, sparse_flow_tensor, k_value=3, voxel_size = None):

    dense_flow = []
    for b_idx in range(len(xyz)):
        
        sparse_xyz = sparse_flow_tensor.coordinates_at(b_idx).cuda() * voxel_size
        sparse_flow = sparse_flow_tensor.features_at(b_idx)

        sqr_dist = pairwise_distance(xyz[b_idx].cuda(), sparse_xyz, normalized=False).squeeze(0)
        sqr_dist, group_idx = torch.topk(sqr_dist, k_value, dim = -1, largest=False, sorted=False)
        

        dist = torch.sqrt(sqr_dist)
        norm = torch.sum(1 / (dist + 1e-7), dim = 1, keepdim = True)
        weight = ((1 / (dist + 1e-7)) / norm ).unsqueeze(-1)

        test = group_idx.reshape(-1)
        sparse_flow = sparse_flow[group_idx.reshape(-1), :].reshape(-1,k_value,3)
        
        dense_flow.append(torch.sum(weight * sparse_flow, dim=1))

    return dense_flow


def upsample_bckg_labels(xyz, sparse_seg_tensor, voxel_size = None):

    upsampled_seg_labels = []
    for b_idx in range(len(xyz)):
        sparse_xyz = sparse_seg_tensor.coordinates_at(b_idx).cuda() * voxel_size
        seg_labels = sparse_seg_tensor.features_at(b_idx)
        sqr_dist = pairwise_distance(xyz[b_idx].cuda(), sparse_xyz, normalized=False).squeeze(0)
        sqr_dist, idx = torch.topk(sqr_dist, 1, dim = -1, largest=False, sorted=False)
        
        
        upsampled_seg_labels.append(seg_labels[idx.reshape(-1)])

    return torch.cat(upsampled_seg_labels,0)



def upsample_cluster_labels(xyz, sparse_seg_tensor, cluster_labels, voxel_size = None):

    upsampled_seg_labels = []

    cluster_labels_all = defaultdict(list)
    for b_idx in range(len(xyz)):
        sparse_xyz = sparse_seg_tensor.coordinates_at(b_idx).cuda() * voxel_size
        seg_labels = sparse_seg_tensor.features_at(b_idx)
        sqr_dist = pairwise_distance(xyz[b_idx].cuda(), sparse_xyz, normalized=False).squeeze(0)
        sqr_dist, idx = torch.topk(sqr_dist, 1, dim = -1, largest=False, sorted=False)
        
        for cluster in cluster_labels[str(b_idx)]:
            cluster_indices = torch.nonzero(idx.reshape(-1)[:,None] == cluster)
        
            cluster_labels_all[str(b_idx)].append(cluster_indices[:,0])

    return cluster_labels_all

