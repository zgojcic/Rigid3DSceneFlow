import os 
import logging 
import torch
import time
import argparse
import yaml 

import numpy as np
from collections import defaultdict
from tqdm import tqdm

import lib.config as config
from lib.utils import n_model_parameters, dict_all_to_device, load_checkpoint
from lib.data import make_data_loader
from lib.logger import prepare_logger



# Set the random seeds for repeatability
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0) 

def main(cfg, logger):
    """
    Main function of this evaluation software. After preparing the data loaders, and the model start with the evaluation process.
    Args:
        cfg (dict): current configuration paramaters
    """

    # Create the output dir if it does not exist 
    if not os.path.exists(cfg['test']['results_dir']):
        os.makedirs(cfg['test']['results_dir'])

    # Get model
    model = config.get_model(cfg)
    device = torch.device('cuda' if (torch.cuda.is_available() and cfg['misc']['use_gpu']) else 'cpu') 

    # Get data loader
    eval_loader = make_data_loader(cfg, phase='test')

    # Log directory
    dataset_name = cfg["data"]["dataset"]

    path2log = os.path.join(cfg['test']['results_dir'], dataset_name, '{}_{}'.format(cfg['method']['backbone'], cfg['misc']['num_points']))

    logger, checkpoint_dir = prepare_logger(cfg, path2log)

    # Output torch and cuda version 
    
    logger.info('Torch version: {}'.format(torch.__version__))
    logger.info('CUDA version: {}'.format(torch.version.cuda))
    logger.info('Starting evaluation of the method {} on {} dataset'.format(cfg['method']['backbone'], dataset_name))

    # Save config file that was used for this experiment
    with open(os.path.join(path2log, "config.yaml"),'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False, allow_unicode=True)


    logger.info("Parameter Count: {:d}".format(n_model_parameters(model)))
    
    # Load the pretrained weights
    if cfg['network']['use_pretrained'] and cfg['network']['pretrained_path']:
        model, optimizer, scheduler, epoch_it, total_it, metric_val_best = load_checkpoint(model, None, None, filename=cfg['network']['pretrained_path'])

    else:
        logger.warning('MODEL RUNS IN EVAL MODE, BUT NO PRETRAINED WEIGHTS WERE LOADED!!!!')


    # Initialize the trainer
    trainer = config.get_trainer(cfg, model,device)

    # if not a pretrained model epoch and iterations should be -1 
    eval_metrics = defaultdict(list)    
    start = time.time()
    
    for it, batch in enumerate(tqdm(eval_loader)):
        # Put all the tensors to the designated device
        dict_all_to_device(batch, device)
        

        metrics = trainer.eval_step(batch)
        
        for key in metrics:
            eval_metrics[key].append(metrics[key])


    stop = time.time()

    # Compute mean values of the evaluation statistics
    result_string = ''

    for key, value in eval_metrics.items():
        if key not in ['true_p', 'true_n', 'false_p', 'false_n']:
            result_string += '{}: {:.3f}; '.format(key, np.mean(value))
    
    if 'true_p' in eval_metrics:
        result_string += '{}: {:.3f}; '.format('dataset_precision_f', (np.sum(eval_metrics['true_p']) / (np.sum(eval_metrics['true_p'])  + np.sum(eval_metrics['false_p'])) ))
        result_string += '{}: {:.3f}; '.format('dataset_recall_f', (np.sum(eval_metrics['true_p']) / (np.sum(eval_metrics['true_p'])  + np.sum(eval_metrics['false_n']))))

        result_string += '{}: {:.3f}; '.format('dataset_precision_b', (np.sum(eval_metrics['true_n']) / (np.sum(eval_metrics['true_n'])  + np.sum(eval_metrics['false_n']))))
        result_string += '{}: {:.3f}; '.format('dataset_recall_b', (np.sum(eval_metrics['true_n']) / (np.sum(eval_metrics['true_n'])  + np.sum(eval_metrics['false_p']))))


    logger.info('Outputing the evaluation metric for: {} {} {} '.format('Flow, ' if cfg['metrics']['flow'] else '', 'Ego-Motion, ' if cfg['metrics']['ego_motion'] else '', 'Bckg. Segmentaion' if cfg['metrics']['semantic'] else ''))
    logger.info(result_string)
    logger.info('Evaluation completed in {}s [{}s per scene]'.format((stop - start), (stop - start)/len(eval_loader)))     


if __name__ == "__main__":
    logger = logging.getLogger


    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()

    cfg = config.get_config(args.config, 'configs/default.yaml')

    main(cfg, logger)