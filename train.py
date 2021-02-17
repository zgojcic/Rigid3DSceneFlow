import os 
import gc
import shutil
import torch
import time
import argparse
import yaml 
import copy
import glob
import numpy as np
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter

import lib.config as config
from lib.utils import n_model_parameters, save_checkpoint, dict_all_to_device, load_checkpoint
from lib.data import make_data_loader
from lib.logger import prepare_logger

# Set the random seeds for repeatability
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0) 

def main(cfg, config_name):
    """
    Main training function: after preparing the data loaders, model, optimizer, and trainer,
    start with the training process.

    Args:
        cfg (dict): current configuration parameters
        config_name (str): path to the config file
    """

    # Create the output dir if it does not exist 
    if not os.path.exists(cfg['misc']['log_dir']):
        os.makedirs(cfg['misc']['log_dir'])

    # Initialize the model
    model = config.get_model(cfg)
    model = model.cuda()

    # Get data loader
    train_loader = make_data_loader(cfg, phase='train')
    val_loader = make_data_loader(cfg, phase='val')

    # Log directory
    dataset_name = cfg["data"]["dataset"]

    now = datetime.now().strftime("%y_%m_%d-%H_%M_%S_%f")
    now += "__Method_" + str(cfg['method']['backbone'])
    now += "__Pretrained_" if cfg['network']['use_pretrained'] and cfg['network']['pretrained_path'] else ''
    if cfg['method']['flow']: now += "__Flow_"
    if cfg['method']['ego_motion']: now += "__Ego_"
    if cfg['method']['semantic']: now += "__Sem_"
    now += "__Rem_Ground_"  if cfg['data']['remove_ground'] else ''
    now += "__VoxSize_" + str(cfg['misc']["voxel_size"])
    now += "__Pts_" + str(cfg['misc']["num_points"])
    path2log = os.path.join(cfg['misc']['log_dir'],"logs_" + dataset_name, now)

    logger, checkpoint_dir = prepare_logger(cfg, path2log)
    tboard_logger = SummaryWriter(path2log)
    
    # Output number of model parameters
    logger.info("Parameter Count: {:d}".format(n_model_parameters(model)))

    # Output torch and cuda version 
    logger.info('Torch version: {}'.format(torch.__version__))
    logger.info('CUDA version: {}'.format(torch.version.cuda))

    # Save config file that was used for this experiment
    with open(os.path.join(path2log, config_name.split(os.sep)[-1]),'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False, allow_unicode=True)

    # Get optimizer and trainer
    optimizer = config.get_optimizer(cfg, model)
    scheduler = config.get_scheduler(cfg, optimizer)


    # Parameters determining the saving and validation interval (if positive denotes iteration if negative epoch)
    stat_interval = cfg['train']['stat_interval']
    stat_interval = stat_interval if stat_interval > 0 else abs(stat_interval* len(train_loader))

    chkpt_interval = cfg['train']['chkpt_interval']
    chkpt_interval = chkpt_interval if chkpt_interval > 0 else abs(chkpt_interval* len(train_loader))

    val_interval = cfg['train']['val_interval']
    val_interval = val_interval if val_interval > 0 else abs(val_interval* len(train_loader))

    # if not a pretrained model epoch and iterations should be -1 
    metric_val_best = np.inf
    running_metrics = {}
    running_losses = {}
    epoch_it = -1
    total_it = -1

    # Load the pretrained weights
    if cfg['network']['use_pretrained'] and cfg['network']['pretrained_path']:
        model, optimizer, scheduler, epoch_it, total_it, metric_val_best = load_checkpoint(model, optimizer, scheduler, filename=cfg['network']['pretrained_path'])

        # Find previous tensorboard files and copy them 
        tb_files = glob.glob(os.sep.join(cfg['network']['pretrained_path'].split(os.sep)[:-1]) + '/events.*')
        for tb_file in tb_files:
            shutil.copy(tb_file, os.path.join(path2log, tb_file.split(os.sep)[-1]))


    # Initialize the trainer
    device = torch.device('cuda' if (torch.cuda.is_available() and cfg['misc']['use_gpu']) else 'cpu') 
    trainer = config.get_trainer(cfg, model, device)
    acc_iter_size = cfg['train']['acc_iter_size']

    # Training loop
    while epoch_it < cfg['train']['max_epoch']: 
        epoch_it += 1
        lr = scheduler.get_last_lr()
        logger.info('Training epoch: {}, LR: {} '.format(epoch_it, lr))
        gc.collect()
        
        train_loader_iter = train_loader.__iter__()
        start = time.time()
        tbar = tqdm(total=len(train_loader) // acc_iter_size, ncols=100)

        for it in range(len(train_loader) // acc_iter_size):
            optimizer.zero_grad()
            total_it += 1
            batch_metrics = {}
            batch_losses = {}


            for iter_idx in range(acc_iter_size):

                batch = train_loader_iter.next()
                
                dict_all_to_device(batch, device)
                losses, metrics, total_loss = trainer.train_step(batch)

                total_loss.backward()

                # Save the running metrics and losses
                if not batch_metrics:
                    batch_metrics = copy.deepcopy(metrics)
                else:
                    for key, value in metrics.items():
                        batch_metrics[key] += value

                if not batch_losses:
                    batch_losses = copy.deepcopy(losses)    
                else:
                    for key, value in losses.items():
                        batch_losses[key] += value


            # Compute the mean value of the metrics and losses of the batch
            for key, value in batch_metrics.items():
                batch_metrics[key] = value / acc_iter_size
                
            for key, value in batch_losses.items():
                batch_losses[key] = value / acc_iter_size

            optimizer.step()
            torch.cuda.empty_cache()
            
            tbar.set_description('Loss: {:.3g}'.format(batch_losses['total_loss']))
            tbar.update(1)

            # Save the running metrics and losses
            if not running_metrics:
                running_metrics = copy.deepcopy(batch_metrics)
            else:
                for key, value in batch_metrics.items():
                    running_metrics[key] += value

            if not running_losses:
                running_losses = copy.deepcopy(batch_losses)    
            else:
                for key, value in batch_losses.items():
                    running_losses[key] += value      
            
            # Logs
            if total_it % stat_interval == stat_interval - 1:
                # Print / save logs
                logger.info("Epoch {0:d} - It. {1:d}: loss = {2:.3f}".format(
                    epoch_it, total_it, running_losses['total_loss'] / stat_interval
                    )
                )

                for key, value in running_losses.items():
                    tboard_logger.add_scalar("Train/{}".format(key), value / stat_interval, total_it)
                    # Reinitialize the values
                    running_losses[key] = 0

                for key, value in running_metrics.items():
                    tboard_logger.add_scalar("Train/{}".format(key), value / stat_interval, total_it)
                    # Reinitialize the values
                    running_metrics[key] = 0

                start = time.time()


            # Run validation
            if total_it % val_interval == val_interval - 1:
                logger.info("Starting the validation")
                val_losses, val_metrics = trainer.validate(val_loader)

                for key, value in val_losses.items():
                    tboard_logger.add_scalar("Val/{}".format(key), value, total_it)


                for key, value in val_metrics.items():
                    tboard_logger.add_scalar("Val/{}".format(key), value, total_it)

                logger.info("VALIDATION -It. {0:d}: total loss: {1:.3f}.".format(total_it, val_losses['total_loss']))


                if val_losses['total_loss'] < metric_val_best:
                    metric_val_best = val_losses['total_loss']
                    logger.info('New best model (loss: {:.4f})'.format(metric_val_best))

                    save_checkpoint(os.path.join(path2log,'model_best.pt'), epoch=epoch_it, it=total_it, model=model,
                                    optimizer=optimizer,scheduler=scheduler,config=cfg, best_val=metric_val_best)
                else:
                    save_checkpoint(os.path.join(path2log,'model_{}.pt'.format(total_it)), epoch=epoch_it, it=total_it, model=model,
                        optimizer=optimizer, scheduler=scheduler, config=cfg, best_val=val_losses['total_loss'])

        # After the epoch if finished update the scheduler
        scheduler.step()

    # Quit after the maximum number of epochs is reached
    logger.info('Training completed after {} Epochs ({} it) with best val metric ({})={}'.format(epoch_it, it, model_selection_metric, metric_val_best))


if __name__ == "__main__":

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()

    # Combine the two config files
    cfg = config.get_config(args.config, 'configs/default.yaml')

    main(cfg, args.config)
