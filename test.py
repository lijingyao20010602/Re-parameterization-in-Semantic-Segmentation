import os
import json
import argparse
import torch
import dataloaders
import models
from utils import losses, setup_logger
from utils.torchsummary import summary
from trainer import Trainer

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def test(args, deploy, logger):
    # DATA LOADERS
    type = 'deploy' if deploy else 'train'
    model_path = os.path.join(args.path,'{}-{}.pth'.format(args.arch, type)) 

    train_loader = get_instance(dataloaders, 'train_loader', args.config)
    val_loader = get_instance(dataloaders, 'val_loader', args.config)

    # MODEL
    test_model = get_instance(models, 'arch', args.config, train_loader.dataset.num_classes, deploy)
    # print(test_model)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        # print(ckpt.keys())
        test_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    # LOSS
    loss = getattr(losses, args.config['loss'])(ignore_index=args.config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=test_model,
        loss=loss,
        resume=None,
        config=args.config,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger)

    trainer.test()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-p', '--path', metavar='PATH', help='path to the weights file')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RepPSP')
    args = parser.parse_args()

    logger = setup_logger(name='Test', output=args.path)
    config_path = os.path.join(args.path,'config.json')
    args.config = json.load(open(config_path))

    logger.info('===============train================')
    test(args, deploy=False, logger=logger)
    logger.info('===============deploy================')
    test(args, deploy=True, logger=logger)
    

