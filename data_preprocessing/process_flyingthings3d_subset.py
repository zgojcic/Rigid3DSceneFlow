import numpy as np
import sys, os
import os.path as osp
from multiprocessing import Pool
import argparse

import IO
from flyingthings3d_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_path', type=str, help="path to the raw data")
parser.add_argument('--save_path', type=str, help="save path")
parser.add_argument('--only_save_near_pts', dest='save_near', action='store_true',
                    help='only save near points to save disk space')

args = parser.parse_args()
root_path = args.raw_data_path
save_path = args.save_path

splits = ['train', 'val']


def process_one_file(params):
    try:
        train_val, fname = params

        save_folder_path = osp.join(save_path, train_val)
        os.makedirs(save_folder_path, exist_ok=True)

        disp1 = IO.read(osp.join(root_path, train_val, 'disparity', 'left', fname + '.pfm'))
        disp1_occ = IO.read(osp.join(root_path, train_val, 'disparity_occlusions', 'left', fname + '.png'))
        disp1_change = IO.read(
            osp.join(root_path, train_val, 'disparity_change', 'left', 'into_future', fname + '.pfm'))
        flow = IO.read(osp.join(root_path, train_val, 'flow', 'left', 'into_future', fname + '.flo'))
        flow_occ = IO.read(osp.join(root_path, train_val, 'flow_occlusions', 'left', 'into_future', fname + '.png'))
        instance_mask = IO.read(osp.join(root_path, train_val, 'object_ids', 'left', fname + '.png'))

        pc1 = pixel2pc(disp1)
        pc2 = next_pixel2pc(flow, disp1 + disp1_change)

        if pc1[..., -1].max() > 0 or pc2[..., -1].max() > 0:
            print('z > 0', train_val, fname, pc1[..., -1].max(), pc1[..., -1].min(), pc2[..., -1].max(),
                  pc2[..., -1].min())

        valid_mask = np.logical_and(disp1_occ == 0, flow_occ == 0)

        pc1 = pc1[valid_mask]
        pc2 = pc2[valid_mask]
        flow = pc2 - pc1
        instance_mask = instance_mask[valid_mask]

        inst_cnt = 0



        if args.save_near:
            near_mask = np.logical_and(pc1[..., -1] > -35., pc2[..., -1] > -35.)
            pc1 = pc1[near_mask]
            pc2 = pc2[near_mask]
            flow = flow[near_mask]
            instance_mask = instance_mask[near_mask]

        # Map instance labels to small numbers
        for inst_label in set(instance_mask.tolist()):
            instance_mask[instance_mask==inst_label] = inst_cnt
            inst_cnt += 1

        if not args.save_near:
            np.savez_compressed(osp.join(save_folder_path,  '{}.npz'.format(fname)), pc1=pc1, pc2=pc2, 
                                                                                     flow=flow, inst_pc1=instance_mask,
                                                                                     inst_pc2=instance_mask)
        else:
            near_mask = np.logical_and(pc1[..., -1] > -35., pc2[..., -1] > -35.)
            np.savez_compressed(osp.join(save_folder_path,  '{}.npz'.format(fname)), pc1=pc1[near_mask],
                                                                                     pc2=pc2[near_mask], flow=flow[near_mask], 
                                                                                     inst_pc1=instance_mask[near_mask],
                                                                                     inst_pc2=instance_mask[near_mask])

        #np.savetxt('test.csv',np.concatenate((pc1,instance_mask.reshape(-1,1)),axis=1))
    except Exception as ex:
        print('error in addressing params', params, 'see exception:')
        print(ex)
        sys.stdout.flush()
        return


if __name__ == '__main__':
    param_list = []
    for train_val in splits:
        tmp_path = osp.join(root_path, train_val, 'disparity_change', 'left', 'into_future')
        param_list.extend([(train_val, item.split('.')[0]) for item in os.listdir(tmp_path)])

    pool = Pool(10)
    pool.map(process_one_file, param_list)
    pool.close()
    pool.join()

    print('Finish all!')
