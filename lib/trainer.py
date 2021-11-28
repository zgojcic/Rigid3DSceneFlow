import torch
import copy
import MinkowskiEngine as ME
from tqdm import tqdm 

from lib.loss import TrainLoss
from lib.metrics import EvalMetrics
from lib.utils import dict_all_to_device

class FlowTrainer:
    ''' 
    Default trainer class for the scene flow training

    Args:
        args (dict): configuration parameters
        model (nn.Module): model
        device (pytorch device)

    ''' 

    def __init__(self, args, model, device):
       
        self.device = device

        self.compute_losses = TrainLoss(args)
        self.compute_metrics = EvalMetrics(args)
        self.model = model.to(device)

    
    def train_step(self, data):
        ''' 
        Performs a single training step.
        
        Args:
            data (dict): input data

        Returns:
            loss_values (dict): all individual loss values
            metric (dict): evaluation metics
            total_loss (torch.tensor): loss value used for training
        
        '''
        
        self.model.train()
        losses, metrics = self._compute_loss_metrics(data)

        # Copy only the loss values not the whole tensors
        loss_values = {}
        for key, value in losses.items():
            loss_values[key] = value.item()
        
        return loss_values, metrics, losses['total_loss']

    
    def eval_step(self, data):
        ''' 
        Performs a single evaluation epoch.
        
        Args:
            data (dict): input data
        
        Returns:
            metric (dict): evaluation metics

        '''
        
        # evaluate model:
        self.model.eval()
        with torch.no_grad():
            _, metrics = self._compute_loss_metrics(data, phase='eval')
       
        return metrics


    def validate(self, val_loader):
        ''' 
        Performs the whole validation 
        
        Args:
            val_loader ( torch data loader): data loader of the validation data            
        '''
        
        # evaluate model:
        self.model.eval()
        running_losses = {}
        running_metrics = {}

        with torch.no_grad():
            for it, batch in enumerate(tqdm(val_loader)):
                
                dict_all_to_device(batch, self.device)
                losses, metrics = self._compute_loss_metrics(batch)

                # Update the running losses
                if not running_losses:
                    running_losses = copy.deepcopy(losses)    
                else:
                    for key, value in losses.items():
                        running_losses[key] += value

                # Update the running metrics
                if not running_metrics:
                    running_metrics = copy.deepcopy(metrics)    
                else:
                    for key, value in metrics.items():
                        running_metrics[key] += value


        for key, value in running_losses.items():
            running_losses[key] = value/len(val_loader)

        for key, value in running_metrics.items():
            running_metrics[key] = value/len(val_loader)

        return running_losses, running_metrics


class MEFlowTrainer(FlowTrainer):
    ''' 
    Trainer class of the 3D rigid scene flow network with ME backbone

    Args:
        args (dict): configuration parameters
        model (nn.Module): model
        device (pytorch device)
    '''

    def __init__(self, args, model, device):

        FlowTrainer.__init__(self, args, model, device)

    def _compute_loss_metrics(self, input_dict, phase='train'):

        ''' 
        Computes the losses and evaluation metrics
        
        Args:
            input_dict (dict): data dictionary

        Return:
            losses (dict): selected loss values
            metric (dict): selected evaluation metric
        '''

        # Run the feature and context encoder
        sinput1 = ME.SparseTensor(features=input_dict['sinput_s_F'].to(self.device),
            coordinates=input_dict['sinput_s_C'].to(self.device))

        sinput2 = ME.SparseTensor(features=input_dict['sinput_t_F'].to(self.device),
            coordinates=input_dict['sinput_t_C'].to(self.device))
                
        if phase == 'train':
            inferred_values = self.model(sinput1, sinput2, input_dict['pcd_s'], input_dict['pcd_t'], input_dict['fg_labels_s'], input_dict['fg_labels_t'])
        else:
            inferred_values = self.model(sinput1, sinput2, input_dict['pcd_eval_s'], input_dict['pcd_eval_t'], input_dict['fg_labels_s'], input_dict['fg_labels_t'])

        losses = self.compute_losses(inferred_values, input_dict)
        
        metrics = self.compute_metrics(inferred_values, input_dict, phase)

        return losses, metrics


    def _demo_step(self, input_dict):

        ''' 
        Runs a short demo and visualizes the output
        
        Args:
            input_dict (dict): data dictionary

        '''

        # Run the feature and context encoder
        sinput1 = ME.SparseTensor(features=input_dict['sinput_s_F'].to(self.device),
            coordinates=input_dict['sinput_s_C'].to(self.device))

        sinput2 = ME.SparseTensor(features=input_dict['sinput_t_F'].to(self.device),
            coordinates=input_dict['sinput_t_C'].to(self.device))
                
        inferred_values = self.model(sinput1, sinput2, input_dict['pcd_eval_s'], input_dict['pcd_eval_t'], input_dict['fg_labels_s'], input_dict['fg_labels_t'])

        
        metrics = self.compute_metrics(inferred_values, input_dict)

        return losses, metrics