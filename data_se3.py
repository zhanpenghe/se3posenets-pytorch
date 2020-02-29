import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import h5py
import pybullet as p
from utils import get_pointcloud


class DataSE3(Dataset):
    def __init__(self, data_path, split, seq_len, direction_num, object_num, select_num=None, use_color=False):
        self.data_path = data_path
        self.seq_len = seq_len
        self.direction_num = direction_num
        self.object_num = object_num
        self.use_color = use_color
        self.idx_list = open(osp.join(self.data_path, '%s.txt' % split)).read().splitlines()
        if select_num is not None:
            self.idx_list = self.idx_list[:select_num]

        self.camera_intr = np.load(osp.join(self.data_path, 'camera_intr.npy'))
        self.camera_pose = np.load(osp.join(self.data_path, 'camera_pose.npy'))
        self.camera_view_matrix = np.load(osp.join(self.data_path, 'camera_view_matrix.npy'))

    def __getitem__(self, index):
        data_dict = {}
        f = h5py.File(osp.join(self.data_path, "%s_%d.hdf5" % (self.idx_list[index // self.seq_len], index % self.seq_len)), "r")

        # init
        init_color_image = np.asarray(f['init']['color_image'], np.int)
        init_depth_image = np.asarray(f['init']['depth_image'])[0]
        init_state = self.get_state(init_color_image, init_depth_image)
        data_dict['init_color_heightmap'] = np.asarray(f['init']['color_heightmap'], dtype=np.float32) / 255
        data_dict['init_color_image'] = np.asarray(f['init']['color_image'], dtype=np.float32) / 255
        data_dict['init_state'] = np.asarray(init_state, dtype=np.float32) #[6, 240, 320] xyz+rgb or [4, 240, 320] xyz+depth

        # action
        data_dict['action'] = self.get_action(f['action'])

        # scene_flow 2d
        data_dict['scene_flow'] = np.transpose(np.asarray(f['next']['scene_flow_2d'], dtype=np.float32), (2, 0, 1)) # [3, 240, 320] xyz

        # next
        next_color_image = np.asarray(f['next']['color_image'], np.int)
        next_depth_image = np.asarray(f['next']['depth_image'])[0]
        next_state = self.get_state(next_color_image, next_depth_image)
        data_dict['next_color_heightmap'] = np.asarray(f['next']['color_heightmap'], dtype=np.float32) / 255
        data_dict['next_color_image'] = np.asarray(f['next']['color_image'], dtype=np.float32) / 255
        data_dict['next_state'] = np.asarray(next_state, dtype=np.float32)  # [6, 240, 320] xyz+rgb or [4, 240, 320] xyz+depth

        return data_dict

    def __len__(self):
        return len(self.idx_list) * self.seq_len


    def get_action(self, action):
        # type vector
        direction, r, c = int(action[0]), int(action[1]), int(action[2])
        if direction < 0:
            direction += self.direction_num
        action_vec = np.zeros(shape=[self.direction_num + 2], dtype=np.float32)
        action_vec[direction] = 1
        action_vec[-2] = r
        action_vec[-1] = c
        return action_vec

    def get_state(self, color_image, depth_image):
        cam_pts, rgb_pts = get_pointcloud(color_image, depth_image, self.camera_intr)
        world_pts = np.transpose(
            np.dot(self.camera_pose[0:3, 0:3], np.transpose(cam_pts)) + np.tile(self.camera_pose[0:3, 3:], (1, cam_pts.shape[0])))
        W, H = depth_image.shape
        world_pts.resize([W, H, 3])
        rgb_pts.resize([W, H, 3])
        if self.use_color:
            state = np.concatenate([world_pts, rgb_pts / 255.0], 2)
        else:
            state = np.concatenate([world_pts, depth_image[..., np.newaxis]], 2)

        state = np.transpose(state, [2, 0, 1])
        return state


if __name__ == '__main__':

    data_path = '../shapenet5-final/'
    data = DataSE3(data_path=data_path, split='train', seq_len=10, direction_num=10, object_num=6, select_num=None, use_color=False)

    import ipdb
    ipdb.set_trace()
    print('Done with loading shapenet data')





