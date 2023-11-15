import argparse
import os
import torch


"""
    一些如文件名之类的参数
    
"""

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="F:", help='Your data dir')
    parser.add_argument('--batch_size', type=int, default=2, help='The number of input images')
    parser.add_argument('--base_lr', type=float, default=0.00001, help='The base learn rate of model')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of times to train the model')
    parser.add_argument('--base_model',type=str,default="SwinV2",help='The basebone of model')
    parser.add_argument('--style',type=str,default='Shinkai',help='Cartoon style')
    parser.add_argument('--gpu_ids',type=int,default=[2,3],help='gpu id')


    return parser


