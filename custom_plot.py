import os
import sys
import numpy as np
import argparse
import open3d as o3d
import time
import torch
from torch.utils.data import dataloader
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval




ROOT_DIR = os.path.dirname(os.path.abspath(__file__))                                                    
 




















sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))                                                     
sys.path.append(os.path.join(ROOT_DIR, 'utils'))                                                         
sys.path.append(os.path.join(ROOT_DIR, 'models'))                                                        
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))                                                       
        
 
from models.graspnet import GraspNet, pred_decode                                                        
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn 
from utils.collision_detector import ModelFreeCollisionDetector   


def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    print("grippers")
    #print(type(grippers[1]))
    #print(grippers[0].is_cpu)

    mesh = o3d.geometry.TriangleMesh()
    for i in range(len(grippers)):
        print(i)    
        o3d.io.write_triangle_mesh("obj_" + str(i).zfill(3) + ".obj", grippers[i])
    #print(f' cloud shape {cloud.shape}')
    #o3d.visualization.draw_geometries([grippers[0]])



# Read two numpy files
def read_numpy_files(file1, file2):
    data1 = np.load(file1)
    data2 = np.load(file2)
    return data1, data2

# Example usage
file1_path = 'test_custom_cloud.npy'
file2_path = './scene_0004/realsense/0000.npy'
cloud, gg_raw = read_numpy_files(file1_path, file2_path)


gg = GraspGroup()
gg.from_npy(file2_path)

print(f" gg {gg}")

vis_grasps(gg, cloud)





