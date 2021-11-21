from __future__ import print_function
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models.util import create_model
from dataset.mini_imagenet import MetaImageNet
from eval.meta_eval import meta_test

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=['resnet12'])
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to .pth model')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.dataset = 'miniImageNet'
    opt.data_root = './data/{}'.format(opt.dataset)
    opt.data_aug = True

    return opt


def main():

    opt = parse_option()

    # test loader
    args = opt
    args.batch_size = args.test_batch_size

    meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False)
    meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                fix_seed=False),
                                batch_size=opt.test_batch_size, shuffle=False, drop_last=False)
    n_cls = 64

    # load model
    # model = create_model(n_cls)
    model = torch.hub.load('pytorch/vision:v0.10.0','resnet18', pretrained = False,num_classes = n_cls)
    ckpt = torch.load(opt.model_path)
    model.load_state_dict(ckpt['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # evalation
    start = time.time()
    val_acc, val_std = meta_test(model, meta_valloader)
    val_time = time.time() - start
    print('val_acc: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc, val_std, val_time))

    # start = time.time()
    # val_acc_feat, val_std_feat = meta_test(model, meta_valloader, use_logit=False)
    # val_time = time.time() - start
    # print('val_acc_feat: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc_feat, val_std_feat, val_time))

    start = time.time()
    test_acc, test_std = meta_test(model, meta_testloader)
    test_time = time.time() - start
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))

    # start = time.time()
    # test_acc_feat, test_std_feat = meta_test(model, meta_testloader, use_logit=False)
    # test_time = time.time() - start
    # print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat, test_std_feat, test_time))


if __name__ == '__main__':
    main()
