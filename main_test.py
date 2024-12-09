import os
import json
import argparse
import time
import pandas as pd
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorboardX import SummaryWriter
from fastprogress import master_bar, progress_bar

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from config import get_config
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from utils.plot_utils import *
from models.gcn_model import ResidualGatedGCNModel
from utils.model_utils import *
import sys
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from diffusion_rl_tsp.utils import TSP_2opt, calculate_distance_matrix2


def parse_args():
    parser = argparse.ArgumentParser(description="Run TSP experiment with GCN model.")
    parser.add_argument('--num_nodes', type=int, default=20, help='Number of nodes in the TSP problem.')
    parser.add_argument('--constraint_type', type=str, default='cluster', help='Number of nodes in the TSP problem.')
    parser.add_argument('--time', type=str, default='1900-01-01')
    parser.add_argument('--beam_size', type=int, default=16)
    
    args = parser.parse_args()
    return args


def metrics_to_str(epoch, time, learning_rate, loss, pred_tour_len, gt_tour_len):
    result = ( 'epoch:{epoch:0>2d}\t'
               'time:{time:.1f}h\t'
               'lr:{learning_rate:.2e}\t'
               'loss:{loss:.4f}\t'
               'pred_tour_len:{pred_tour_len:.3f}\t'
               'gt_tour_len:{gt_tour_len:.3f}'.format(
                   epoch=epoch,
                   time=time/3600,
                   learning_rate=learning_rate,
                   loss=loss,
                   pred_tour_len=pred_tour_len,
                   gt_tour_len=gt_tour_len))
    return result


def test(net, config, master_bar, mode='test', constraint_type='basic', constraint=None):
    net.eval()

    # Assign parameters
    num_nodes = config.num_nodes
    num_neighbors = config.num_neighbors
    batch_size = config.batch_size
    batches_per_epoch = config.batches_per_epoch
    config.beam_size = args.beam_size
    beam_size = config.beam_size
    
    date_per_type = {
        'basic': '',
        'box': '240710',
        'path': '240711',
        'cluster': '240721',
    }
    if config.constraint_type == 'basic':
        config.file_name = f'./data/tsp{config.num_nodes}_test_concorde.txt'
    else:
        config.file_name = f'./data/tsp{config.num_nodes}_{config.constraint_type}_constraint_{date_per_type.get(config.constraint_type)}.txt'
    val_filepath = config.file_name
    test_filepath = config.file_name

    # Load TSP data
    if mode == 'val':
        dataset = GoogleTSPReader(num_nodes, num_neighbors, batch_size=batch_size, filepath=val_filepath)
    elif mode == 'test':
        dataset = GoogleTSPReader(num_nodes, num_neighbors, batch_size=batch_size, filepath=test_filepath)
    batches_per_epoch = dataset.max_iter
    dataset = iter(dataset)
    
    edge_cw = None
    running_loss = 0.0
    running_pred_tour_len = 0.0
    running_gt_tour_len = 0.0
    running_nb_data = 0
    running_nb_batch = 0
    pointss = None
    pred_tours = None
    gt_tours = None
    gt_tour_lens = []
    basic_costs = []
    gt_costs = []
    penalty_counts = []

    with torch.no_grad():
        start_test = time.time()
        
        for batch_num in progress_bar(range(batches_per_epoch), parent=master_bar):
            try:
                batch = next(dataset)
            except StopIteration:
                break

            x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
            x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
            x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
            x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
            y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
            # y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)
            
            if constraint_type == 'box':
                intersection_matrices = np.zeros((batch_size, num_nodes, num_nodes))
                for batch_idx in range(batch_size):
                    points = batch.points[batch_idx]
                    constraint = batch.constraint[batch_idx]
                    _, intersection_matrix = calculate_distance_matrix2(points, constraint)
                    intersection_matrices[batch_idx] = intersection_matrix
                constraint = intersection_matrices
                
            else:
                constraint = batch.constraint

            if type(edge_cw) != torch.Tensor:
                edge_labels = y_edges.cpu().numpy().flatten()
                edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

            y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
            loss = loss.mean()  # Take mean of loss across multiple GPUs

            # Get batch beamsearch tour prediction
            if mode == 'val':  # Validation: faster 'vanilla' beamsearch
                bs_nodes = beamsearch_tour_nodes(
                    y_preds, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='logits', constraint_type=constraint_type, constraint = constraint)
            elif mode == 'test':  # Testing: beamsearch with shortest tour heuristic 
                bs_nodes = beamsearch_tour_nodes_shortest(
                    y_preds, x_edges_values, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='logits', constraint_type=constraint_type, constraint = constraint)
            
            if pred_tours is None:
                pred_tours = bs_nodes
                gt_tours = batch.gt_tour
            else:
                pred_tours = torch.cat((pred_tours, bs_nodes), dim=0)
                gt_tours = np.concatenate((gt_tours, batch.gt_tour), axis=0)
                
            if mode == 'test':
                for i in range(batch_size):
                    solver = TSP_2opt(points = batch.points[i], constraint_type = constraint_type, constraint = constraint[i])
                    pred_tour = bs_nodes[i].tolist() + bs_nodes[i].tolist()[:1]
                    basic_cost = solver.evaluate(pred_tour)
                    gt_cost = solver.evaluate(batch.gt_tour[i]-1)
                    penalty_count = solver.count_constraints(pred_tour)
                    
                    basic_costs.append(basic_cost)
                    gt_costs.append(gt_cost)
                    penalty_counts.append(penalty_count)
            
            pred_tour_len = mean_tour_len_nodes(x_edges_values, bs_nodes)
            gt_tour_len = np.mean(batch.tour_len)
            gt_tour_lens.extend(batch.tour_len)
            
            running_nb_data += batch_size
            running_loss += batch_size* loss.data.item()
            running_pred_tour_len += batch_size* pred_tour_len
            running_gt_tour_len += batch_size* gt_tour_len
            running_nb_batch += 1
            
            
            # Log intermediate statistics
            result = ('loss:{loss:.4f} pred_tour_len:{pred_tour_len:.3f} gt_tour_len:{gt_tour_len:.3f}'.format(
                loss=running_loss/running_nb_data,
                pred_tour_len=running_pred_tour_len/running_nb_data,
                gt_tour_len=running_gt_tour_len/running_nb_data))
            master_bar.child.comment = result

    loss = running_loss / running_nb_data
    pred_tour_len = running_pred_tour_len / running_nb_data
    gt_tour_len = running_gt_tour_len / running_nb_data

    return time.time() - start_test, loss, pred_tour_len, gt_tour_len, pred_tours.detach().cpu().numpy(), gt_tours, basic_costs, gt_costs, penalty_counts

if __name__ == '__main__':
    # now = time.strftime('%y%m%d_%H%M%S')
    # Parse arguments
    args = parse_args()
    config_path = f'/mnt/home/zuwang/workspace/graph-convnet-tsp/configs/tsp{args.num_nodes}.json'
    config = get_config(config_path)
    config.num_nodes = args.num_nodes
    config.constraint_type = args.constraint_type
    config.result_file_name = f'tsp{config.num_nodes}_gnn_beamsize{args.beam_size}_{args.time}'
    print("Loaded {}:\n{}".format(config_path, json.dumps(config, indent=4)))

    ## Configure GPU options
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    if torch.cuda.is_available():
        print("CUDA available, using GPU ID {}".format(config.gpu_id))
        dtypeFloat = torch.float32 #torch.cuda.FloatTensor
        dtypeLong = torch.int64 #torch.cuda.LongTensor
        torch.cuda.manual_seed(1)
    else:
        print("CUDA not available")
        dtypeFloat = torch.float32 #torch.FloatTensor
        dtypeLong = torch.int64 #torch.LongTensor
        torch.manual_seed(1)

    torch.autograd.set_detect_anomaly(True)

    net = nn.DataParallel(ResidualGatedGCNModel(config, dtypeFloat, dtypeLong))
    if torch.cuda.is_available():
        net.cuda()

    learning_rate = config.learning_rate
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    log_dir = f"./logs/{config.expt_name}/"
    if torch.cuda.is_available():
        checkpoint = torch.load(log_dir+"best_val_checkpoint.tar")
    else:
        checkpoint = torch.load(log_dir+"best_val_checkpoint.tar", map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # train_loss = checkpoint['train_loss']
    # val_loss = checkpoint['val_loss']
    # for param_group in optimizer.param_groups:
    #     learning_rate = param_group['lr']
    print(f"Loaded checkpoint from epoch {epoch}")

    epoch_bar = master_bar(range(epoch + 1, epoch + 2))
    config.batch_size = 128

    for epoch in epoch_bar:
        # Set validation dataset as the test dataset so that we can perform 
        # greedy and vanilla beam search on test data without hassle!
        # config.val_filepath = config.test_filepath
        
        # Greedy search
        # config.beam_size = 1
        # t=time.time()
        # val_time, val_loss, val_pred_tour_len, val_gt_tour_len = test(net, config, epoch_bar, mode='val', constraint_type = config.constraint_type)
        # print("G time: {}s".format(time.time()-t))
        # epoch_bar.write('G: ' + metrics_to_str(epoch, val_time, learning_rate, val_loss, val_pred_tour_len, val_gt_tour_len))
        
        # Vanilla beam search
        # config.beam_size = 1280
        # t=time.time()
        # val_time, val_loss, val_pred_tour_len, val_gt_tour_len = test(net, config, epoch_bar, mode='val', constraint_type = config.constraint_type)
        # print("BS time: {}s".format(time.time()-t))
        # epoch_bar.write('BS: ' + metrics_to_str(epoch, val_time, learning_rate, val_loss, val_pred_tour_len, val_gt_tour_len))
        
        # Beam search with shortest tour heuristic
        config.beam_size = 1280
        # t = time.time()
        test_time, test_loss, pred_tour_len, gt_tour_len, pred_tours, gt_tours, basic_costs, gt_costs, penalty_counts = test(net, config, epoch_bar, mode='test', constraint_type=config.constraint_type)
        # print("BS* time: {}s".format(time.time()-t))
        # epoch_bar.write('BS*: ' + metrics_to_str(epoch, test_time, learning_rate, test_loss, test_pred_tour_len, test_gt_tour_len))
        
    print('total time : ', test_time)
    result_df = pd.DataFrame(
        {
            'sample_idx' : range(len(pred_tours)),
            'basic_cost' : basic_costs,
            'gt_cost' : gt_costs,
            'penalty_count' : penalty_counts,
            'pred_tour' : [str(row) for row in pred_tours],
            'gt_tour' : [str(row) for row in gt_tours[:,:-1]-1],
        }
    )

    os.makedirs(f'./Results/{config.constraint_type}/gnn', exist_ok=True)
    os.makedirs(f'../diffusion_rl_tsp/Results/{config.constraint_type}/gnn', exist_ok=True)

    result_df.to_csv(f'./Results/{config.constraint_type}/gnn/{config.result_file_name}.csv', index=False)    
    result_df.to_csv(f'../diffusion_rl_tsp/Results/{config.constraint_type}/gnn/{config.result_file_name}.csv', index=False)