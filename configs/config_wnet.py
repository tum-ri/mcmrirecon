import os
import time

import sys
sys.path.append("..")
from utils.utils import DotDict

# import utils.util as util


""" Configuration parameters """

config = {
    'data_root': '/home/md/Documents/MasterSemester3/PracticalCourse/NeuralNets/Data',  # Root path to /Train and /Val folder
    'model_save_path': '/home/md/Documents/MasterSemester3/PracticalCourse/NeuralNets/Data/SavedNetworks/', # path to where networks are saved and also resumed from
    'mask_URate': 'R5', # R5 or R10
    'challenge': 'multicoil',  # Type of challenge ('singlecoil' or 'multicoil')
    'dim': (218, 170),  # Dimension of input kspace
    'domain': 'ki',  # Input output domain ('ii', 'ik', 'ki', 'kk')
    'sample_rate': 1,  # Sample rate of input data
    'sampling_dist': 'poisson',  # Sampling distribution undersampling ('gaussian', 'uniform', 'possion')
    'norm': True,  # Compute the Root Sum of Squares (RSS)
    'sum_of_squares': False,
    'slice_cut_train': (50, 50),  # Crop (first and last) slices during read in
    'slice_cut_val': (50, 50),
    'num_edge_slices': 0,  # denotes number of edge slices for the edge model. 0 means vanilla slice cut! --> no edge/middle model
    'edge_model': True,  # denotes whether to train on edges or on middle part
    'batch_size': 4,  # Batch size
    'num_workers': 4, # Denotes the number of processes that generate batches in parallel
    'dense': True,   # activates dense
    'architecture': 'ii', # architecture of the network
    'lossFunction': 'mse+vif', # loss function of the network
    'lr': 1e-4, # learning rate
    'complex_weight_init': False,
    'num_epochs': 100, # number of epochs the network trains
    'mask_flag': True, # whether to apply mask for DC or just add orginial undersampled kspace
    'complex_net': False,
    'verbose_delay': 30, #at what epoch to start computing metrics of validate set (saves time)
    'verbose_gap': 50, #number of iterations inbetween metric calculations of training set
    'resume': None, #denote checkpoint to load to resume training. None if you want to train from scratch
    }
