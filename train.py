import os
import json
import argparse
import torch
import dataloaders
import models
import inspect
import math
from utils import losses
from utils import setup_logger, Logger
from utils.torchsummary import summary
from trainer import Trainer

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, args):

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)

    # MODEL
    if config['arch']['type'].startswith('Rep'): 
        model = get_instance(models, 'arch', config, train_loader.dataset.num_classes, False)
    else:
        model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=args.resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=Logger(),
        outputdir = args.outputdir)

    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str, help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-t', '--trial', default=None, type=str, help='idx of experiment')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if config['arch']['type'].startswith('Rep'): 
        args.outputdir = config['arch']['args']['repconv'] + '_' + args.trial
    elif args.trial == '0':
        args.outputdir = 'test'
    else:
        args.outputdir = None

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args)
