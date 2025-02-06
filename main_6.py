# -*- coding: utf-8 -*-
# @Time    : 3/11/2024 6:54 PM
# @Author  : Gang Qu
# @FileName: main_5.py
# GNN classification experiment using 1 IQ scores
import argparse
import utils
import os
from os.path import join as pj
from os.path import abspath
import json
import datasets
import torch
from models.load_model import load_model
import time
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import scripts.utils.utils as utils
from scripts.train.train import train_epoch, evaluate_network
from scripts.datasets.HCDP_dataset import HCDP_DGL
from models.gradcam import GNN_GradCAM
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
def main(args):
    manifold_thre = 0.05  # set to avoid oversmooth
    buckets = {"nih_totalcogcomp_ageadjusted": [(60, 81), (129, 150)]}
    now = str(time.strftime("%Y_%m_%d_%H_%M"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set random seed for reproducing results
    if args.seed:
        utils.seed_it(args.seed)

    # import config files and update args
    config_file = pj(abspath('configs'), args.config + '.json')
    with open(config_file) as f:
        config = json.load(f)
    config['dataset'] = args.dataset
    if args.dropout:
        config['net_params']['dropout'] = args.dropout
    if args.x_attributes is None:
        args.x_attributes = ['FC', 'SC', 'CT']

    save_name = now + config['dataset'] + '_' + config['model_save_suffix'] + '_' + config[
        'model'] + '_' + args.task + '_' + str(args.sparse);
    if not os.path.exists(abspath('results/hcdp')):
        os.makedirs(abspath('results/hcdp'))
    if not os.path.exists(abspath('results/hcdp/pretrained')):
        os.makedirs(abspath('results/hcdp/pretrained'))
    if not os.path.exists(abspath('results/hcdp/loggers')):
        os.makedirs(abspath('results/hcdp/loggers'))

    # print the config and args information
    logger = utils.get_logger(name=save_name, path='results/loggers')
    logger.info(args)
    logger.info(config)

    # define tensorboard for visualization of the training
    if not os.path.exists(abspath('results/runs')):
        os.makedirs(abspath('results/runs'))
    writer = SummaryWriter(log_dir=pj(abspath('results/runs'), save_name), flush_secs=30)

    # define dataset

    dataset = HCDP_DGL(task=args.task, x_attributes=args.x_attributes, y_attribute=args.y_attribute,
                       buckets_dict=buckets, attributes_to_group=args.attributes_group)

    train_size = int(args.train_val_test[0] * len(dataset))
    val_size = int(args.train_val_test[1] * len(dataset))
    val_size = 0
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    if 'CT' in args.x_attributes:
        additional_features = []
        for idx in train_set.indices:  # Access the original indices from the subset
            _, g, _ = dataset[idx]  # Use the original dataset to access the data
            additional_features.append(g.ndata['additional_feature'])

        # Concatenate all additional features from the training set and compute mean and std
        all_additional_features = torch.cat(additional_features, dim=0)
        mean = all_additional_features.mean(dim=0)
        std = all_additional_features.std(dim=0)
        dataset.mean = mean
        dataset.std = std


    model_save_dir = pj(abspath('results'), save_name + '.pth')

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.batch_size,
                                               collate_fn=utils.collate_dgl)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, collate_fn=utils.collate_dgl)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                              collate_fn=utils.collate_dgl)

    print(dataset[0])
    ##
    logger.info('#' * 60)
    # define the model
    model = load_model(config)
    model.to(device)
    logger.info(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2)
    if config['pretrain']:
        checkpoint = torch.load(abspath(pj('results', config['pretrain_model_name'])))
        model.load_state_dict(checkpoint['model'])
        print('Using pretrained model: ', config['pretrain_model_name'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    min_test_loss = 1e12
    #
    # early_stopping = utils.EarlyStopping(tolerance=10, min_delta=0)
    ##
    for epoch in range(args.max_epochs):
        epoch_loss_train, optimizer, gs = train_epoch(model, optimizer, device,
                                                      data_loader=train_loader, epoch=epoch, task_type=args.task,
                                                      logger=logger, writer=writer,
                                                      weight_score=args.weight_score, alpha=args.alpha, beta=args.beta,
                                                      l_l=args.ll, l_m=args.lm)
        # epoch_loss_val, _ = evaluate_network(model, device, val_loader, epoch, task_type=args.task, logger=logger,
        #                                      phase='Val', writer=writer, weight_score=args.weight_score)
        epoch_loss_test, _ = evaluate_network(model, device, test_loader, epoch, task_type=args.task, logger=logger,
                                              phase='Test', writer=writer, weight_score=args.weight_score)
        scheduler.step(epoch_loss_train)

        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalars(main_tag='Loss/epoch_losses',
                           tag_scalar_dict={'Train': epoch_loss_train},
                           global_step=epoch)

        # scheduler.step(epoch_loss_train)
        epoch_loss_test = epoch_loss_train
        if epoch_loss_test < min_test_loss:
            min_test_loss = epoch_loss_test
            # model_state = copy.deepcopy(model.state_dict())
            # optimizer_state = copy.deepcopy(optimizer.state_dict())
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

        # early_stopping(epoch_loss_train, epoch_loss_val)
        # if early_stopping.early_stop and gs.pl < manifold_thre:
        #     logger.info("We are at epoch: {}".format(epoch))
        #     break
    writer.close()
    torch.save(checkpoint, model_save_dir)

    # symmetric_mask = model.layer0.edge_weight.cpu().detach().numpy()
    #
    # # Save the mask to a file
    # mask_filename = r'F:\projects\AiyingT1MultimodalFusion\scripts\results\hcdp\cam\symmetric_mask_totalIQ.npy'
    # np.save(mask_filename, symmetric_mask)
    #
    #
    # all_activations = []
    # all_gradients = []
    # grad_cam_util = GNN_GradCAM(model)
    #
    # for iter, sample_i in enumerate(train_loader):
    #     subject_ids, gs, batch_targets = sample_i
    #     batch_targets = batch_targets.to(device)
    #     batch_scores, gs = model(gs.to(device), gs.ndata['x'].to(device))
    #     activations_0, gradients_0 = grad_cam_util.generate_grad_cam(gs, gs.ndata['x'], target_layer_index=1,
    #                                                                  task_type='classification', target_class=0)
    #     activations_1, gradients_1 = grad_cam_util.generate_grad_cam(gs, gs.ndata['x'], target_layer_index=1,
    #                                                                  task_type='classification', target_class=1)
    #     all_activations.append([activations_0.cpu().detach().numpy(), activations_1.cpu().detach().numpy()])
    #     all_gradients.append([gradients_0.cpu().detach().numpy(), gradients_1.cpu().detach().numpy()])
    #
    # all_activations = np.array(all_activations)
    # all_gradients = np.array(all_gradients)
    # print(all_activations.shape, all_gradients.shape)
    # np.save(r'F:\projects\AiyingT1MultimodalFusion\scripts\results\hcdp\cam\activations_totalIQ_coup.npy', all_activations)
    # np.save(r'F:\projects\AiyingT1MultimodalFusion\scripts\results\hcdp\cam\gradients_totalIQ_coup.npy', all_gradients)





if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='HCDP classification')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--max_epochs', default=10, type=int, help='max number of epochs')
    parser.add_argument('--L2', default=1e-7, type=float, help='L2 regularization')
    parser.add_argument('--dropout', default=0, help='dropout rate')
    parser.add_argument('--seed', default=70, type=int, help='random seed')
    parser.add_argument('--config', default='MASKGCN_classification_IQ', help='config file name')
    parser.add_argument('--task', default='classification', choices=['regression', 'classification'],
                        help='task type')

    parser.add_argument('--x_attributes', action='append', default=None, help='modalities')
    parser.add_argument('--y_attribute', default=['nih_totalcogcomp_ageadjusted'],
                        help='target attribute')
    parser.add_argument('--attributes_group', default=["nih_totalcogcomp_ageadjusted"], help='attributes to group')
    parser.add_argument('--train_val_test', default=[0.7, 0.1, 0.2], help='train, val, test split')
    parser.add_argument('--dataset', default='dglHCP', help='dataset name')
    parser.add_argument('--sparse', default=30, type=int, help='sparsity for knn graph')
    parser.add_argument('--gl', default=False, help='graph learning beta')
    parser.add_argument('--weight_score', default=[0.5, 0.5], help='weight score')
    parser.add_argument('--alpha', type=float, default=1e-6, help='L1 sparsity for mask')
    parser.add_argument('--beta', type=float, default=1e-6, help='L2 sparsity for mask')
    parser.add_argument('--ll', type=float, default=1e-6, help='penaly for laplacian')
    parser.add_argument('--lm', type=float, default=1e-2, help='penaly for mask')
    args = parser.parse_args()

    main(args)