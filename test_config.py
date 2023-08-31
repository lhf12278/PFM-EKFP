import argparse
from config import log_config 
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='command for evaluate on CUHK-PEDES')
    # Directory
    parser.add_argument('--dataset', type=str, default='CUHK-PEDES', help='CUHK_PEDES, ICFG-PEDES, RSTPReid')
    parser.add_argument('--image_dir', type=str, default='./data/CUHK-PEDES/imgs',help='directory to store dataset')
    parser.add_argument('--anno_dir', type=str, default='./data/project_directory/cuhkpedes/processed_data',help='directory to store anno')
    
    parser.add_argument('--model_path', type=str, default='./checkpoints/data/model_data',help='directory to load checkpoint')
    parser.add_argument('--log_dir', type=str, default='./Person_search/data/logs',help='directory to store log')
    parser.add_argument('--pretrain_dir', type=str, default='./data/project_directory/pretrained_models/imagenet21k+imagenet2012_ViT-B_16.npz',
                        help='the path of vit parameters')
    parser.add_argument('--resnet50_dir', type=str, default=None,
                        help='the path of vit parameters')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument('--feature_size', type=int, default=768)
    #parser.add_argument('--num_heads', type=int, default=12, help='#num of heads')
    # Default setting
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--epoch_start', type=int)
    parser.add_argument('--checkpoint_dir', type=str,default='./checkpoints/data/model_data' )

 

    args = parser.parse_args()
    return args



def config():
    args = parse_args()
    log_config(args, 'test')
    return args
