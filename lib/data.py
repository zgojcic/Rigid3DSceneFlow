import os
import torch 
import logging 

import numpy as np
import torch.utils.data as data
import MinkowskiEngine as ME


def to_tensor(x):
    if isinstance(x, torch.Tensor):
      return x
    elif isinstance(x, np.ndarray):
      return torch.from_numpy(x)
    else:
      raise ValueError("Can not convert to torch tensor {}".format(x))
    
def collate_fn(list_data):
    pc_1,pc_2, coords1, coords2, feats1, feats2, fg_labels_1, \
    fg_labels_2, flow, R_ego, t_ego, pc_eval_1, pc_eval_2, flow_eval, fg_labels_eval_1, fg_labels_eval_2 = list(zip(*list_data))

    pc_batch1, pc_batch2 = [], []
    pc_eval_batch1, pc_eval_batch2 = [], []
    fg_labels_batch1, fg_labels_batch2 = [], []
    fg_labels_eval_batch1, fg_labels_eval_batch2 = [], []
    R_ego_batch, t_ego_batch = [],[]
    flow_batch, flow_eval_batch, len_batch = [], [], []
    batch_id = 0

    for batch_id, _ in enumerate(coords1):
        N1 = coords1[batch_id].shape[0]
        N2 = coords2[batch_id].shape[0]
        len_batch.append([N1, N2])

        pc_batch1.append(to_tensor(pc_1[batch_id]).float())
        pc_batch2.append(to_tensor(pc_2[batch_id]).float())

        pc_eval_batch1.append(to_tensor(pc_eval_1[batch_id]).float())
        pc_eval_batch2.append(to_tensor(pc_eval_2[batch_id]).float())

        fg_labels_batch1.append(to_tensor(fg_labels_1[batch_id]))
        fg_labels_batch2.append(to_tensor(fg_labels_2[batch_id]))

        fg_labels_eval_batch1.append(to_tensor(fg_labels_eval_1[batch_id]))
        fg_labels_eval_batch2.append(to_tensor(fg_labels_eval_2[batch_id]))

        R_ego_batch.append(to_tensor(R_ego[batch_id]).unsqueeze(0))
        t_ego_batch.append(to_tensor(t_ego[batch_id]).unsqueeze(0))

        flow_batch.append(to_tensor(flow[batch_id]))
        flow_eval_batch.append(to_tensor(flow_eval[batch_id]))

    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords=coords1, feats=feats1)
    coords_batch2, feats_batch2 = ME.utils.sparse_collate(coords=coords2, feats=feats2)
  

    # Concatenate all lists
    fg_labels_batch1 = torch.cat(fg_labels_batch1, 0).long()
    fg_labels_batch2 = torch.cat(fg_labels_batch2, 0).long()
    flow_batch = torch.cat(flow_batch, 0).float()
    flow_eval_batch = torch.cat(flow_eval_batch, 0).float()
    R_ego_batch = torch.cat(R_ego_batch, 0).float()
    t_ego_batch = torch.cat(t_ego_batch, 0).float()
    fg_labels_eval_batch1 = torch.cat(fg_labels_eval_batch1, 0).long()
    fg_labels_eval_batch2 = torch.cat(fg_labels_eval_batch2, 0).long()

    return {
        'pcd_s': pc_batch1,
        'pcd_t': pc_batch2,
        'sinput_s_C': coords_batch1,
        'sinput_s_F': feats_batch1.float(),
        'sinput_t_C': coords_batch2,
        'sinput_t_F': feats_batch2.float(),
        'fg_labels_s': fg_labels_batch1,
        'fg_labels_t': fg_labels_batch2,
        'flow': flow_batch,
        'R_ego': R_ego_batch,
        't_ego': t_ego_batch,
        'pcd_eval_s': pc_eval_batch1,
        'pcd_eval_t': pc_eval_batch2,
        'flow_eval': flow_eval_batch,
        'fg_labels_eval_s': fg_labels_eval_batch1,
        'fg_labels_eval_t': fg_labels_eval_batch2,
        'len_batch': len_batch
    }


class MELidarDataset(data.Dataset):
    def __init__(self, phase, config):
        
        self.files = []
        self.root = config['data']['root']
        self.config = config
        self.input_features = config['data']['input_features']
        self.num_points = config['misc']['num_points']
        self.voxel_size = config['misc']['voxel_size']
        self.remove_ground = True if (config['data']['remove_ground'] and config['data']['dataset'] in ['StereoKITTI_ME','LidarKITTI_ME','SemanticKITTI_ME','WaymoOpen_ME']) else False
        self.dataset = config['data']['dataset']
        self.only_near_points = config['data']['only_near_points']
        self.phase = phase
    
        self.randng = np.random.RandomState()
        self.device = torch.device('cuda' if (torch.cuda.is_available() and config['misc']['use_gpu']) else 'cpu') 

        self.augment_data = config['data']['augment_data']

        logging.info("Loading the subset {} from {}".format(phase,self.root))

        subset_names = open(self.DATA_FILES[phase]).read().split()

        for name in subset_names:
            self.files.append(name)

    def __getitem__(self, idx):
        file = os.path.join(self.root,self.files[idx])
        file_name = file.replace(os.sep,'/').split('/')[-1]
        
        # Load the data
        data = np.load(file)
        pc_1 = data['pc1']
        pc_2 = data['pc2']

        if 'pose_s' in data:
            pose_1 = data['pose_s']
        else:
            pose_1 = np.eye(4)

        if 'pose_t' in data:
            pose_2 = data['pose_t']
        else:
            pose_2 = np.eye(4)

        if 'sem_label_s' in data:
            labels_1 = data['sem_label_s']
        else:
            labels_1 = np.zeros(pc_1.shape[0])


        if 'sem_label_t' in data:
            labels_2 = data['sem_label_t']
        else:
            labels_2 = np.zeros(pc_2.shape[0])

        if 'flow' in data:
            flow = data['flow']
        else:
            flow = np.zeros_like(pc_1)

        # Remove the ground and far away points
        # In stereoKITTI the direct correspondences are provided therefore we remove,
        # if either of the points fullfills the condition (as in hplflownet, flot, ...)

        if self.dataset in ["SemanticKITTI_ME", 'LidarKITTI_ME', "WaymoOpen_ME"]:
            if self.remove_ground:
                if self.phase == 'test':
                    is_not_ground_s = (pc_1[:, 1] > -1.4)
                    is_not_ground_t = (pc_2[:, 1] > -1.4)

                    pc_1 = pc_1[is_not_ground_s,:]
                    labels_1 = labels_1[is_not_ground_s]
                    flow = flow[is_not_ground_s,:]

                    pc_2 = pc_2[is_not_ground_t,:]
                    labels_2 = labels_2[is_not_ground_t]

                # In the training phase we randomly select if the ground should be removed or not 
                elif np.random.rand() > 1/4:
                    is_not_ground_s = (pc_1[:, 1] > -1.4)
                    is_not_ground_t = (pc_2[:, 1] > -1.4)

                    pc_1 = pc_1[is_not_ground_s,:]
                    labels_1 = labels_1[is_not_ground_s]
                    flow = flow[is_not_ground_s,:]

                    pc_2 = pc_2[is_not_ground_t,:]
                    labels_2 = labels_2[is_not_ground_t]

            if self.only_near_points:
                is_near_s = (pc_1[:, 2] < 35)
                is_near_t = (pc_2[:, 2] < 35)

                pc_1 = pc_1[is_near_s,:]
                labels_1 = labels_1[is_near_s]
                flow = flow[is_near_s,:]

                pc_2 = pc_2[is_near_t,:]
                labels_2 = labels_2[is_near_t]

        else:
            if self.remove_ground:
                is_not_ground = np.logical_not(np.logical_and(pc_1[:, 1] < -1.4, pc_2[:, 1] < -1.4))
                pc_1 = pc_1[is_not_ground,:]
                pc_2 = pc_2[is_not_ground,:]
                flow = flow[is_not_ground,:]

            if self.only_near_points:
                is_near = np.logical_and(pc_1[:, 2] < 35, pc_1[:, 2] < 35)
                pc_1 = pc_1[is_near,:]
                pc_2 = pc_2[is_near,:]
                flow = flow[is_near,:]

        # Augment the point cloud by randomly rotating and translating them (recompute the ego-motion if augmention is applied!)
        if self.augment_data and self.phase != 'test':
            T_1 = np.eye(4)
            T_2 = np.eye(4)

            T_1[0:3,3] = (np.random.rand(3) - 0.5) * 0.5
            T_2[0:3,3] = (np.random.rand(3) - 0.5) * 0.5

            T_1[1,3] = (np.random.rand(1) - 0.5) * 0.1 
            T_2[1,3] = (np.random.rand(1) - 0.5) * 0.1

            pc_1 = (np.matmul(T_1[0:3, 0:3], pc_1.transpose()) + T_1[0:3,3:4]).transpose()
            pc_2 = (np.matmul(T_2[0:3, 0:3], pc_2.transpose()) + T_2[0:3,3:4]).transpose()

            pose_1 = np.matmul(pose_1, np.linalg.inv(T_1))
            pose_2 = np.matmul(pose_2, np.linalg.inv(T_2))

            rel_trans = np.linalg.inv(pose_2) @ pose_1

            R_ego = rel_trans[0:3,0:3]
            t_ego = rel_trans[0:3,3:4]
        else:
            # Compute relative pose that transform the point from the source point cloud to the target
            rel_trans = np.linalg.inv(pose_2) @ pose_1
            R_ego = rel_trans[0:3,0:3]
            t_ego = rel_trans[0:3,3:4]


        # Sample n points for evaluation before the voxelization
        # If less than desired points are available just consider the maximum
        if pc_1.shape[0] > self.num_points:
            idx_1 = np.random.choice(pc_1.shape[0], self.num_points, replace=False)
        else:
            idx_1 = np.random.choice(pc_1.shape[0], pc_1.shape[0], replace=False)

        if pc_2.shape[0] > self.num_points:
            idx_2 = np.random.choice(pc_2.shape[0], self.num_points, replace=False)
        else:
            idx_2 = np.random.choice(pc_2.shape[0], pc_2.shape[0], replace=False)

        pc_1_eval = pc_1[idx_1,:]
        flow_eval = flow[idx_1,:]
        labels_1_eval = labels_1[idx_1]

        pc_2_eval = pc_2[idx_2,:]
        labels_2_eval = labels_2[idx_2]

        # Voxelization
        _, sel1 = ME.utils.sparse_quantize(np.ascontiguousarray(pc_1) / self.voxel_size, return_index=True)
        _, sel2 = ME.utils.sparse_quantize(np.ascontiguousarray(pc_2) / self.voxel_size, return_index=True)


        # Slect the voxelized points
        pc_1 = pc_1[sel1,:]
        labels_1 = labels_1[sel1]
        flow = flow[sel1,:]

        pc_2 = pc_2[sel2,:]
        labels_2 = labels_2[sel2]

        # If more voxels then the selected number of points are remaining randomly sample them
        if pc_1.shape[0] > self.num_points:
            idx_1 = np.random.choice(pc_1.shape[0], self.num_points, replace=False)
        else:
            idx_1 = np.random.choice(pc_1.shape[0], pc_1.shape[0], replace=False)

        if pc_2.shape[0] > self.num_points:
            idx_2 = np.random.choice(pc_2.shape[0], self.num_points, replace=False)
        else:
            idx_2 = np.random.choice(pc_2.shape[0], pc_2.shape[0], replace=False)

        pc_1 = pc_1[idx_1,:]
        labels_1 = labels_1[idx_1]
        flow = flow[idx_1,:]

        pc_2 = pc_2[idx_2,:]
        labels_2 = labels_2[idx_2]


        # Get sparse indices
        coords1 = np.floor(pc_1 / self.voxel_size)
        coords2 = np.floor(pc_2 / self.voxel_size)


        feats_train1, feats_train2 = [], []

        if self.input_features == 'occupancy':
            feats_train1.append(np.ones((pc_1.shape[0], 1)))
            feats_train2.append(np.ones((pc_2.shape[0], 1)))

        elif self.input_features == 'absolute_coords':
            feats_train1.append(pc_1)
            feats_train2.append(pc_2)

        elif self.input_features == 'relative_coords':
            feats_train1.append(pc_1 - (coords1 * self.voxel_size))
            feats_train2.append(pc_2 - (coords2 * self.voxel_size))

        else:
            raise ValueError('{} not recognized as a valid input feature!'.format(self.input_features))

        feats1 = np.hstack(feats_train1)
        feats2 = np.hstack(feats_train2)

        # Foreground points (class label bellow 40 or above 99 -> binary label 1)
        fg_labels_1 = np.zeros((labels_1.shape[0]))
        fg_labels_1[((labels_1 < 40) | (labels_1 > 99))] = 1
        fg_labels_1[labels_1 == 0] = -1

        fg_labels_2 = np.zeros((labels_2.shape[0]))
        fg_labels_2[((labels_2 < 40) | (labels_2 > 99))] = 1
        fg_labels_2[labels_2 == 0] = -1

        fg_labels_1_eval = np.zeros((labels_1_eval.shape[0]))
        fg_labels_1_eval[((labels_1_eval < 40) | (labels_1_eval > 99))] = 1
        fg_labels_1_eval[labels_1_eval == 0] = -1

        fg_labels_2_eval = np.zeros((labels_2_eval.shape[0]))
        fg_labels_2_eval[((labels_2_eval < 40) | (labels_2_eval > 99))] = 1
        fg_labels_2_eval[labels_2_eval == 0] = -1

        return (pc_1, pc_2, coords1, coords2, feats1, feats2, fg_labels_1, fg_labels_2, flow,
                R_ego, t_ego, pc_1_eval, pc_2_eval, flow_eval, fg_labels_1_eval, fg_labels_2_eval)

    def __len__(self):
        return len(self.files)

    def reset_seed(self,seed=41):
        logging.info('Resetting the data loader seed to {}'.format(seed))
        self.randng.seed(seed)


class FlyingThings3D_ME(MELidarDataset):
    # 3D Match dataset all files
    DATA_FILES = {
        'train': './configs/datasets/flying_things_3d/train.txt',
        'val': './configs/datasets/flying_things_3d/val.txt',
        'test': './configs/datasets/flying_things_3d/test.txt'
    }

class StereoKITTI_ME(MELidarDataset):
    # 3D Match dataset all files
    DATA_FILES = {
        'train': './configs/datasets/stereo_kitti/test.txt',
        'val': './configs/datasets/stereo_kitti/test.txt',
        'test': './configs/datasets/stereo_kitti/test.txt'
    }

class SemanticKITTI_ME(MELidarDataset):
    # 3D Match dataset all files
    DATA_FILES = {
        'train': './configs/datasets/semantic_kitti/train.txt',
        'val': './configs/datasets/semantic_kitti/val.txt',
        'test': './configs/datasets/semantic_kitti/val.txt'
    }

class LidarKITTI_ME(MELidarDataset):
    # 3D Match dataset all files
    DATA_FILES = {
        'train': './configs/datasets/lidar_kitti/test.txt',
        'val': './configs/datasets/lidar_kitti/test.txt',
        'test': './configs/datasets/lidar_kitti/test.txt'
    }


class WaymoOpen_ME(MELidarDataset):
    # 3D Match dataset all files
    DATA_FILES = {
        'train': './configs/datasets/waymo_open/train.txt',
        'val': './configs/datasets/waymo_open/val.txt',
        'test': './configs/datasets/waymo_open/test.txt'
    }


# Map the datasets to string names
ALL_DATASETS = [FlyingThings3D_ME, StereoKITTI_ME, SemanticKITTI_ME, LidarKITTI_ME, WaymoOpen_ME]

dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, neighborhood_limits=None, shuffle_dataset=None):
    """
    Defines the data loader based on the parameters specified in the config file
    Args:
        config (dict): dictionary of the arguments
        phase (str): phase for which the data loader should be initialized in [train,val,test]
        shuffle_dataset (bool): shuffle the dataset or not
    Returns:
        loader (torch data loader): data loader that handles loading the data to the model
    """

    assert config['misc']['run_mode'] in ['train','val','test']

    if shuffle_dataset is None:
        shuffle_dataset = config['misc']['run_mode'] != 'test'

    # Select the defined dataset
    Dataset = dataset_str_mapping[config['data']['dataset']]

    dset = Dataset(phase, config=config)

    drop_last = False if config['misc']['run_mode'] == 'test' else True

    loader = torch.utils.data.DataLoader(
                dset,
                batch_size=config[phase]['batch_size'],
                shuffle=shuffle_dataset,
                num_workers=config[phase]['num_workers'],
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=drop_last
            )

    return loader