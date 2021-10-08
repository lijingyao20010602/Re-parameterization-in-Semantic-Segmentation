import argparse
import os
import json
import torch
import dataloaders
import models
import copy

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def set_args():
    parser = argparse.ArgumentParser(description='RepPSP Conversion')
    parser.add_argument('-p', '--path', metavar='PATH', help='path to the weights file')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RepPSP')
    args = parser.parse_args()
    args.deploy=True

    args.load = os.path.join(args.path,'{}-train.pth'.format(args.arch))
    args.save = os.path.join(args.path,'{}-deploy.pth'.format(args.arch))

    config_path = os.path.join(args.path,'config.json')
    args.config = json.load(open(config_path))
    train_loader = get_instance(dataloaders, 'train_loader', args.config)
    args.n_cls = train_loader.dataset.num_classes
    return args


def model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)

    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
        for m in module.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
            for j in m.modules():
                if hasattr(j, 'switch_to_deploy'):
                    j.switch_to_deploy()
                for k in j.modules():
                    if hasattr(k, 'switch_to_deploy'):
                        k.switch_to_deploy()


    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


def convert():
    args = set_args()
    
    train_model = get_instance(models, 'arch', args.config, args.n_cls, False)

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        # print(ckpt.keys())
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    model_convert(train_model, save_path=args.save)
    print('{} is converted successfully'.format(args.save))


if __name__ == '__main__':
    convert()