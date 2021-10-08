import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from RepPSP import RepPSP, RepPSPDense

def set_args():
    parser = argparse.ArgumentParser(description='RepPSP Conversion')
    parser.add_argument('load', metavar='LOAD', help='path to the weights file')
    parser.add_argument('save', metavar='SAVE', help='path to the weights file')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RepPSP')
    return parser.parse_args()


def model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


def convert():
    args = set_args()

    train_model = RepPSP(deploy=False)

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        print(ckpt.keys())
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    model_convert(train_model, save_path=args.save)


if __name__ == '__main__':
    convert()