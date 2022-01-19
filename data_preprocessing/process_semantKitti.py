import os 
import glob
import argparse
import re
import copy

import open3d as o3d
import numpy as np 
from multiprocessing import Pool

# Some of the functions are taken from pykitti https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))

def load_poses(file):
    """Load and parse ground truth poses"""
    tmp_poses = np.genfromtxt(file, delimiter=' ').reshape(-1,3,4)
    poses = np.repeat(np.expand_dims(np.eye(4),0), tmp_poses.shape[0], axis=0)
    poses[:,0:3,:] = tmp_poses
    return poses

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

# This part of the code is taken from the semanticKITTI API

def open_label(filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
        raise TypeError("Filename should be string type, "
                        "but was {type}".format(type=str(type(filename))))

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))

    return label

def set_label(label, points):
    """ Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
        raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == points.shape[0]:
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16    # instance id in upper half
    else:
        print("Points shape: ", points.shape)
        print("Label shape: ", label.shape)
        raise ValueError("Scan and Label don't contain same number of points")

    # sanity check
    assert((sem_label + (inst_label << 16) == label).all())

    return sem_label, inst_label





def transform_point_cloud(x1, R, t):
    """
    Transforms the point cloud using the giver transformation paramaters
    
    Args:
        x1  (np array): points of the point cloud [n,3]
        R   (np array): estimated rotation matrice [3,3]
        t   (np array): estimated translation vectors [3,1]
    Returns:
        x1_t (np array): points of the transformed point clouds [n,3]
    """
    x1_t = (np.matmul(R, x1.transpose()) + t).transpose()

    return x1_t

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def sorted_alphanum(file_list_ordered):
    """
    Sorts the list alphanumerically
    Args:
        file_list_ordered (list): list of files to be sorted
    Return:
        sorted_list (list): input list sorted alphanumerically
    """
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    sorted_list = sorted(file_list_ordered, key=alphanum_key)

    return sorted_list

def get_file_list(path, extension=None):
    """
    Build a list of all the files in the provided path
    Args:
        path (str): path to the directory 
        extension (str): only return files with this extension
    Return:
        file_list (list): list of all the files (with the provided extension) sorted alphanumerically
    """
    if extension is None:
        file_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        file_list = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)

    return file_list


def get_folder_list(path):
    """
    Build a list of all the files in the provided path
    Args:
        path (str): path to the directory 
        extension (str): only return files with this extension
    Returns:
        file_list (list): list of all the files (with the provided extension) sorted alphanumerically
    """
    folder_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    folder_list = sorted_alphanum(folder_list)
    
    return folder_list


def extract_moving_objects(save_path, frame_idx, pts, sem_label, inst_label, moving_threshold = 100):
    """
    Extracts the point belonging to individual moving objects and saves them to a file 
    Args:
        save_path (str): path where to save the files
        frame_idx (str): current frame number
        pts (np.array): point cloud of the source frame
        sem_label (np.array): semantic labels
        inst_label (np.array): temporally consistent instance labels
        moving_threshold (int): label above which the classes denote moving objects

    Returns:
    
    """
    moving_idx_s = np.where(sem_label >= 100)[0]
    
    # Filter out the points and labels
    sem_label = sem_label[moving_idx_s]
    inst_label = inst_label[moving_idx_s]
    pts = pts[moving_idx_s,:]

    # Unique semantic labels
    unique_labels = np.unique(sem_label)

    pcd = o3d.geometry.PointCloud()
    for label in unique_labels:
        class_idx = np.where(sem_label == label)[0]
        class_instances = inst_label[class_idx]
        class_points = pts[class_idx,:]
        tmp_instances = np.unique(class_instances)

        for instance in tmp_instances:
            object_idx = np.where(class_instances == instance)[0]
            object_points = class_points[object_idx, :]
        
            # Save the points and sample a random color
            object_color = np.repeat(np.random.random(size=3).reshape(1,-1),repeats=object_points.shape[0], axis=0)
            pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd.colors = o3d.utility.Vector3dVector(object_color)

            if not os.path.exists(os.path.join(save_path, 'objects', '{}_{}'.format(label, instance))):
                os.makedirs(os.path.join(save_path, 'objects', '{}_{}'.format(label, instance)))


            # Save point in the npz and ply format
            np.savez(os.path.join(save_path, 'objects', '{}_{}'.format(label, instance),'{}.npz'.format(frame_idx)),
                        pts=object_points)
            
            o3d.io.write_point_cloud(os.path.join(save_path, 'objects', '{}_{}'.format(label, instance),
                                        '{}.ply'.format(frame_idx)), pcd)



class semanticKITTIProcesor:
    def __init__(self, args):
        self.root_path = args.raw_data_path
        self.save_path = args.save_path
        self.save_ply = args.save_ply
        self.save_near = args.save_near
        self.n_processes = args.n_processes

        self.scenes = get_folder_list(self.root_path)

    def run_processing(self):

        if self.n_processes < 1:
            self.n_processes = 1

        pool = Pool(self.n_processes)
        pool.map(self.process_scene, self.scenes)
        pool.close()
        pool.join()

    def process_scene(self, scene):
        scene_name = scene.split(os.sep)[-1]

        # Create a save file if not existing
        if not os.path.exists(os.path.join(self.save_path, scene_name)):
            os.makedirs(os.path.join(self.save_path, scene_name))
        
        # Load transformation paramters
        poses = load_poses(os.path.join(scene,'poses.txt'))
        tr_velo_cam = read_calib_file(os.path.join(scene,'calib.txt'))['Tr'].reshape(3,4)
        tr_velo_cam = np.concatenate((tr_velo_cam,np.array([0,0,0,1]).reshape(1,4)),axis=0)
        frames = get_file_list(os.path.join(scene,'velodyne'), extension='.bin')

        if os.path.isdir(os.path.join(scene,'labels')):
            labels = get_file_list(os.path.join(scene,'labels'), extension='.label')
            test_scene = False
                    
            assert len(frames) == len(labels), "Number of point cloud fils and label files is not the same!!"
        
        else:
            test_scene = True




        for idx in range(len(frames)-1):
            frame_name_s = frames[idx].split(os.sep)[-1].split('.')[0]
            frame_name_t = frames[idx + 1].split(os.sep)[-1].split('.')[0]

            pc_s = load_velo_scan(frames[idx])[:,:3]
            pc_t = load_velo_scan(frames[idx + 1])[:,:3]

            # Transform both point cloud to the camera coordinate system (check KITTI webpage)
            pc_s = transform_point_cloud(pc_s, tr_velo_cam[:3, :3], tr_velo_cam[:3, 3:4])
            pc_t = transform_point_cloud(pc_t, tr_velo_cam[:3, :3], tr_velo_cam[:3, 3:4])

            # Rotate 180 degrees around z axis (to be in accordance to KITTI flow as used by other datsets)
            pc_s[:,0], pc_s[:,1] = -pc_s[:,0], -pc_s[:,1]
            pc_t[:,0], pc_t[:,1] = -pc_t[:,0], -pc_t[:,1]


            

            if not test_scene:
                # Load the labels
                sem_label_s, inst_label_s = set_label(open_label(labels[idx]), pc_s)
                sem_label_t, inst_label_t = set_label(open_label(labels[idx + 1]), pc_t)

                # Filter out points which are behind the car (to be in accordance with the stereo datasets)
                front_mask_s = pc_s[:,2] > 1.5
                front_mask_t = pc_t[:,2] > 1.5
                pc_s = pc_s[front_mask_s, :]
                pc_t = pc_t[front_mask_t,:] 

                sem_label_s = sem_label_s[front_mask_s]
                inst_label_s = inst_label_s[front_mask_s]

                sem_label_t = sem_label_t[front_mask_t]
                inst_label_t = inst_label_t[front_mask_t]

                if self.save_near:
                    near_mask_s = pc_s[:,2] < 35
                    near_mask_t = pc_t[:,2] < 35
                    pc_s = pc_s[near_mask_s, :]
                    pc_t = pc_t[near_mask_t,:] 

                    sem_label_s = sem_label_s[near_mask_s]
                    inst_label_s = inst_label_s[near_mask_s]

                    sem_label_t = sem_label_t[near_mask_t]
                    inst_label_t = inst_label_t[near_mask_t]

                # Extract the stable parts (sem. labels above 99 denote moving objects)
                # Could also remove 11, 15, 30, 31, 32 (classes like cyclist, person, ...)
                # Motion labels are 1 if moving and 0 if stable
                stable_idx_s = np.where(sem_label_s < 100)[0]
                stable_idx_t = np.where(sem_label_t < 100)[0]
                mot_label_s = np.ones_like(sem_label_s)
                mot_label_s[stable_idx_s] = 0

                mot_label_t = np.ones_like(sem_label_t)
                mot_label_t[stable_idx_t] = 0
                

                # Extract ego motion from the gt poses
                T_st = np.matmul(poses[idx,:,:],np.linalg.inv(poses[idx + 1,:,:]))



                np.savez_compressed(os.path.join(self.save_path, scene_name, '{}_{}.npz'.format(frame_name_s, frame_name_t)), 
                                                                        pc1=pc_s, 
                                                                        pc2=pc_t, 
                                                                        sem_label_s=sem_label_s,
                                                                        sem_label_t=sem_label_t,
                                                                        inst_label_s=inst_label_s,
                                                                        inst_label_t=inst_label_t,
                                                                        mot_label_s=mot_label_s,
                                                                        mot_label_t=mot_label_t,
                                                                        pose_s=poses[idx,:,:],
                                                                        pose_t=poses[idx + 1,:,:])
            else:
                # Filter out points which are behind the car (to be in accordance with the stereo datasets)
                front_mask_s = pc_s[:,2] > 1.5
                front_mask_t = pc_t[:,2] > 1.5
                pc_s = pc_s[front_mask_s, :]
                pc_t = pc_t[front_mask_t,:] 

                if self.save_near:
                    near_mask_s = pc_s[:,2] < 35
                    near_mask_t = pc_t[:,2] < 35
                    pc_s = pc_s[near_mask_s, :]
                    pc_t = pc_t[near_mask_t,:] 

                np.savez_compressed(os.path.join(self.save_path, scene_name, '{}_{}.npz'.format(frame_name_s, frame_name_t)), 
                                                        pc1=pc_s, 
                                                        pc2=pc_t,
                                                        pose_s=poses[idx,:,:],
                                                        pose_t=poses[idx + 1,:,:])

            
            # Save point clouds as ply files
            if self.save_ply:
                pcd_s = o3d.geometry.PointCloud()
                pcd_t = o3d.geometry.PointCloud()
                pcd_s.points = o3d.utility.Vector3dVector(pc_s)
                pcd_t.points = o3d.utility.Vector3dVector(pc_t)

                o3d.io.write_point_cloud(os.path.join(self.save_path, scene_name, '{}.ply'.format(frame_name_s)), pcd_s)
                o3d.io.write_point_cloud(os.path.join(self.save_path, scene_name, '{}.ply'.format(frame_name_t)), pcd_t)




# Define and process command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--raw_data_path", type=str, default="test", help='path to the raw files')
parser.add_argument('--save_path', type=str, help="save path")
parser.add_argument('--n_processes', type=int, default=10,
                    help='number of processes used for multi-processing')
parser.add_argument('--save_ply', action='store_true',
                    help='save point clouds also in ply format')
parser.add_argument('--save_near', action='store_true',
                    help='only save near points (less than 35m)')


args = parser.parse_args()


processor = semanticKITTIProcesor(args)

processor.run_processing()