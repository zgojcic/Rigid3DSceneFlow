import torch
import torch.nn as nn

from lib.utils import compute_epe, rotation_error, translation_error, precision_at_one, evaluate_binary_class

class EvalMetrics(nn.Module):
    """
    Computes all the evaluation metric used to either monitor the training process or evaluate the method
    
    Args:
       args: parameters controling the initialization of the evaluation metrics

    """

    def __init__(self, args):
        nn.Module.__init__(self)


        self.args = args
        self.device = torch.device('cuda' if (torch.cuda.is_available() and args['misc']['use_gpu']) else 'cpu') 

    def __call__(self, inferred_values, gt_data):
        
        # Initialize the dictionary
        metrics = {}
        
        if (self.args['method']['flow'] and self.args['metrics']['flow']):
            assert (('refined_flow' in inferred_values) & ('flow_eval' in gt_data)), "Flow metrics selected \
                                but either est or gt flow not provided"
            
            # Compute the end point error of the flow vectors

            # If bg/fg labels are available use them to also compute f-EPE and b-EPE
            if 'fg_labels_eval_s' in gt_data and self.args['data']['dataset'] not in ["FlyingThings3D_ME", "StereoKITTI_ME"]:
                ego_metrics = compute_epe(inferred_values['refined_rigid_flow'], gt_data['flow_eval'], sem_label=gt_data['fg_labels_eval_s'], eval_stats=True)
            else:
                ego_metrics = compute_epe(inferred_values['refined_rigid_flow'], gt_data['flow_eval'], eval_stats =True)
            
            for key, value in ego_metrics.items():
                metrics[key] = value

        # Compute the ego-motion metric
        if self.args['method']['ego_motion'] and self.args['metrics']['ego_motion']:
            assert (('R_est' in inferred_values) & ('R_ego' in gt_data)), "Ego motion metric selected \
                                            but either est or gt ego motion not provided"

            r_error = rotation_error(inferred_values['R_est'], gt_data['R_ego'])
            
            metrics['mean_r_error'] = torch.mean(r_error).item()
            metrics['max_r_error'] = torch.max(r_error).item()
            metrics['min_r_error'] = torch.min(r_error).item()

            t_error = translation_error(inferred_values['t_est'], gt_data['t_ego'])
            
            metrics['mean_t_error'] = torch.mean(t_error).item()
            metrics['max_t_error'] = torch.max(t_error).item()
            metrics['min_t_error'] = torch.min(t_error).item()


        # Compute the background segmentation metric
        if self.args['method']['semantic'] and self.args['metrics']['semantic']:
            assert (('semantic_logits_s_all' in inferred_values) & ('fg_labels_eval_s' in gt_data)), "Background segmentation metric selected \
                                            but either est or gt labels not provided"
                                            
            pred_label = inferred_values['semantic_logits_s_all'].max(1)[1]
            pre_f, pre_b, rec_f, rec_b = precision_at_one(pred_label, gt_data['fg_labels_eval_s'])

            metrics['precision_f'] = pre_f.item()
            metrics['recall_f'] = rec_f.item()
            metrics['precision_b'] = pre_b.item()
            metrics['recall_b'] = rec_b.item()


            true_p, true_n, false_p, false_n = evaluate_binary_class(pred_label, gt_data['fg_labels_eval_s'])

            metrics['true_p'] = true_p.item()
            metrics['true_n'] = true_n.item()
            metrics['false_p'] = false_p.item()
            metrics['false_n'] = false_n.item()

        return metrics