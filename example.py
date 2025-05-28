import os
import sys
import numpy as np
import argparse
import time
import torch
import json 
from PIL import Image
from torch.utils.data import DataLoader
#from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))


from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--depth_path', help='depth image path', default=None, required=True)
parser.add_argument('--mask_path', help='mask image path', default=None, required=True)
parser.add_argument('--camera_path', help='camera instrinsics', default=None, required=True)
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None, required=True)
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default=None, required=True)
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--num_points', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
cfgs = parser.parse_args()




## LOAD image and masks  and other stats

## convert them to pointcloud 

## make them to input bacth state 

def read_data_from_text(cfgs,factor_depth_=1.0,image_height=480.0,image_width=640.0):
        
        depth_path = cfgs.depth_path
        mask_path = cfgs.mask_path
        camera_path = cfgs.camera_path
        depth = np.array(Image.open(depth_path)) # depth data
        seg = np.array(Image.open(mask_path)) # segmentation mask , check what all values does it have, is bibnay for object not object or is it 
        
        with open(cfgs.camera_path, 'r') as f:
            data = json.load(f)

        intrinsic = np.array(data)

        factor_depth = factor_depth_

        camera = CameraInfo(image_width, image_height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        mask = depth_mask
        seg_mask = (seg > 0)


        mask = (depth_mask & seg_mask)
        cloud_masked = cloud[mask]

        if len(cloud_masked) >= cfgs.num_points:
            idxs = np.random.choice(len(cloud_masked), cfgs.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), cfgs.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    }

        return minkowski_collate_fn([ret_dict])


def run_inf(cfgs):

    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
        
    net.eval()

    data = read_data_from_text(cfgs)
    for key in data:
        data[key] = torch.tensor(data[key]).to(device)
    
    end_points = net(data)

    return end_points

if __name__ == "__main__":

    end_points = run_inf(cfgs)

    print(end_points)

