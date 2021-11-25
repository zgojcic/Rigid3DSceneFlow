import torch
import torch.nn as nn

from lib.utils import transform_point_cloud, kabsch_transformation_estimation
from utils.chamfer_distance import ChamferDistance


class TrainLoss(nn.Module):
    """
    Training loss consists of a ego-motion loss, background segmentation loss, and a foreground loss. 
    The l1 flow loss is used for the full supervised experiments only. 

    Args:
       args: parameters controling the initialization of the loss functions

    """

    def __init__(self, args):
        nn.Module.__init__(self)


        self.args = args
        self.device = torch.device('cuda' if (torch.cuda.is_available() and args['misc']['use_gpu']) else 'cpu') 

        # Flow loss
        self.flow_criterion = nn.L1Loss(reduction='mean')

        # Ego motion loss
        self.ego_l1_criterion = nn.L1Loss(reduction='mean')
        self.ego_outlier_criterion = OutlierLoss()
        
        # Background segmentation loss
        if args['loss']['background_loss'] == 'weighted':

            # Based on the dataset analysis there are 14 times more background labels
            seg_weight = torch.tensor([1.0, 20.0]).to(self.device)
            self.seg_criterion = torch.nn.CrossEntropyLoss(weight=seg_weight, ignore_index=-1)
        
        else:
            self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Foreground loss
        self.chamfer_criterion = ChamferDistance()
        self.rigidity_criterion = nn.L1Loss(reduction='mean')

    def __call__(self, inferred_values, gt_data):
        
        # Initialize the dictionary
        losses = {}
        
        if self.args['method']['flow'] and self.args['loss']['flow_loss']:
            assert (('coarse_flow' in inferred_values) & ('flow' in gt_data)), 'Flow loss selected \
                                                                    but either est or gt flow not provided'

            losses['refined_flow_loss'] = self.flow_criterion(inferred_values['refined_flow'], 
                                                gt_data['flow']) * self.args['loss'].get('flow_loss_w', 1.0)

            losses['coarse_flow_loss'] = self.flow_criterion(inferred_values['coarse_flow'], 
                                                 gt_data['flow']) * self.args['loss'].get('flow_loss_w', 1.0)


        if self.args['method']['ego_motion'] and self.args['loss']['ego_loss']:
            assert (('R_est' in inferred_values) & ('R_s_t' in gt_data) is not None), "Ego motion loss selected \
                                            but either est or gt ego motion not provided"
                                                            
            assert 'permutation' in inferred_values is not None, 'Outlier loss selected \
                                                                        but the permutation matrix is not provided'

            # Only evaluate on the background points
            mask = (gt_data['fg_labels_s'] == 0)

            prev_idx = 0
            pc_t_gt, pc_t_est = [], []

            # Iterate over the samples in the batch
            for batch_idx in range(gt_data['R_ego'].shape[0]):
                
                # Convert the voxel indices back to the coordinates
                p_s_temp = gt_data['sinput_s_C'][prev_idx: prev_idx + gt_data['len_batch'][batch_idx][0],:].to(self.device) * self.args['misc']['voxel_size']
                mask_temp = mask[prev_idx: prev_idx + gt_data['len_batch'][batch_idx][0]]

                # Transform the point cloud with gt and estimated ego-motion parameters
                pc_t_gt_temp = transform_point_cloud(p_s_temp[mask_temp,1:4], gt_data['R_ego'][batch_idx,:,:], gt_data['t_ego'][batch_idx,:,:])
                pc_t_est_temp = transform_point_cloud(p_s_temp[mask_temp,1:4], inferred_values['R_est'][batch_idx,:,:], inferred_values['t_est'][batch_idx,:,:])
                
                pc_t_gt.append(pc_t_gt_temp.squeeze(0))
                pc_t_est.append(pc_t_est_temp.squeeze(0))
                
                prev_idx += gt_data['len_batch'][batch_idx][0]

            pc_t_est = torch.cat(pc_t_est, 0)
            pc_t_gt = torch.cat(pc_t_gt, 0)

            losses['ego_loss'] = self.ego_l1_criterion(pc_t_est, pc_t_gt) * self.args['loss'].get('ego_loss_w', 1.0)
            losses['outlier_loss'] = self.ego_outlier_criterion(inferred_values['permutation']) * self.args['loss'].get('inlier_loss_w', 1.0)

        # Background segmentation loss
        if self.args['method']['semantic'] and self.args['loss']['background_loss']:
            assert (('semantic_logits_s' in inferred_values) & ('fg_labels_s' in gt_data)), "Background loss selected but either est or gt labels not provided"
            
            semantic_loss = torch.tensor(0.0).to(self.device)

            semantic_loss += self.seg_criterion(inferred_values['semantic_logits_s'].F, gt_data['fg_labels_s']) * self.args['loss'].get('bg_loss_w', 1.0)

            # If the background labels for the target point cloud are available also use them for the loss computation
            if 'semantic_logits_t' in inferred_values:
                semantic_loss += self.seg_criterion(inferred_values['semantic_logits_t'].F, gt_data['fg_labels_t']) * self.args['loss'].get('bg_loss_w', 1.0)
                semantic_loss = semantic_loss/2

            losses['semantic_loss'] = semantic_loss

        # Foreground loss
        if self.args['method']['clustering'] and self.args['loss']['foreground_loss']:
            assert ('clusters_s' in inferred_values), "Foreground loss selected but inferred cluster labels not provided"
            
            rigidity_loss = torch.tensor(0.0).to(self.device)
            chamfer_loss = torch.tensor(0.0).to(self.device)

            xyz_s = torch.cat(gt_data['pcd_s'], 0).to(self.device)
            xyz_t = torch.cat(gt_data['pcd_t'], 0).to(self.device)

            # Two-way chamfer distance for the foreground points (only compute if both point clouds have more than 50 foreground points)
            if torch.where(gt_data['fg_labels_s'] == 1)[0].shape[0] > 50 and torch.where(gt_data['fg_labels_t'] == 1)[0].shape[0] > 50:

                foreground_mask_s = (gt_data['fg_labels_s'] == 1)
                foreground_mask_t = (gt_data['fg_labels_t'] == 1)

                foreground_xyz_s = xyz_s[foreground_mask_s,:]
                foreground_flow = inferred_values['refined_rigid_flow'][foreground_mask_s,:]
                foreground_xyz_t = xyz_t[foreground_mask_t,:]

                dist1, dist2 = self.chamfer_criterion(foreground_xyz_t.unsqueeze(0), (foreground_xyz_s + foreground_flow).unsqueeze(0))

                # Clamp the distance to prevent outliers (objects that appear and disappear from the scene)
                dist1 = torch.clamp(torch.sqrt(dist1), max=1.0)
                dist2 = torch.clamp(torch.sqrt(dist2), max=1.0)

                chamfer_loss += ((torch.mean(dist1) + torch.mean(dist2)) / 2.0)

            losses['chamfer_loss'] = chamfer_loss* self.args['loss'].get('cd_loss_w', 1.0)


            # Rigidity loss (flow vectors of each cluster should be congruent)
            n_clusters = 0
            # Iterate over the clusters and enforce rigidity within each cluster
            for batch_idx in inferred_values['clusters_s']:
        
                for cluster in inferred_values['clusters_s'][batch_idx]:
                    cluster_xyz_s = xyz_s[cluster,:].unsqueeze(0)
                    cluster_flow = inferred_values['refined_rigid_flow'][cluster,:].unsqueeze(0)
                    reconstructed_xyz = cluster_xyz_s + cluster_flow

                    # Compute the unweighted Kabsch estimation (transformation parameters which best explain the vectors)
                    R_cluster, t_cluster, _, _ = kabsch_transformation_estimation(cluster_xyz_s, reconstructed_xyz)

                    # Detach the gradients such that they do not flow through the tansformation parameters but only through flow
                    rigid_xyz = (torch.matmul(R_cluster, cluster_xyz_s.transpose(1, 2)) + t_cluster ).detach().squeeze(0).transpose(0,1)
                    
                    rigidity_loss += self.rigidity_criterion(reconstructed_xyz.squeeze(0), rigid_xyz)

                    n_clusters += 1

            n_clusters = 1.0 if n_clusters == 0 else n_clusters            
            losses['rigidity_loss'] = (rigidity_loss / n_clusters) * self.args['loss'].get('rigid_loss_w', 1.0)

        # Compute the total loss as the sum of individual losses
        total_loss = 0.0
        for key in losses:
            total_loss += losses[key]

        losses['total_loss'] = total_loss
        return losses 






class OutlierLoss():
    """
    Outlier loss used regularize the training of the ego-motion. Aims to prevent Sinkhorn algorithm to 
    assign to much mass to the slack row and column.

    """
    def __init__(self):

        self.reduction = 'mean'

    def __call__(self, perm_matrix):

        ref_outliers_strength = []
        src_outliers_strength = []

        for batch_idx in range(len(perm_matrix)):
            ref_outliers_strength.append(1.0 - torch.sum(perm_matrix[batch_idx], dim=1))
            src_outliers_strength.append(1.0 - torch.sum(perm_matrix[batch_idx], dim=2))

        ref_outliers_strength = torch.cat(ref_outliers_strength,1)
        src_outliers_strength = torch.cat(src_outliers_strength,0)

        if self.reduction.lower() == 'mean':
            return torch.mean(ref_outliers_strength) + torch.mean(src_outliers_strength)
        
        elif self.reduction.lower() == 'none':
            return  torch.mean(ref_outliers_strength, dim=1) + \
                                             torch.mean(src_outliers_strength, dim=1)
