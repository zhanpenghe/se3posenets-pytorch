import os.path as osp
import collections
import math
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np
import dominate
from dominate.tags import *
from skimage import measure
import torch


# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotm(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotm2euler(R):
    assert (isRotm(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def angle2rotm(angle_axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke
    angle = angle_axis[0]
    axis = angle_axis[1:]

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[0.0, -axis[2], axis[1]],
                   [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]], dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M[:3, :3]


def transform_points(pts, transform):
    # pts = [3xN] array
    # transform: [3x4]
    pts_t = np.dot(transform[0:3, 0:3], pts) + np.tile(transform[0:3, 3:], (1, pts.shape[1]))
    return pts_t


def get_pointcloud(color_img, depth_img, camera_intrinsics):
    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x, pix_y = np.meshgrid(np.linspace(0, im_w - 1, im_w), np.linspace(0, im_h - 1, im_h))
    cam_pts_x = np.multiply(pix_x - camera_intrinsics[0, 2], depth_img / camera_intrinsics[0, 0])
    cam_pts_y = np.multiply(pix_y - camera_intrinsics[1, 2], depth_img / camera_intrinsics[1, 1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h * im_w, 1)
    cam_pts_y.shape = (im_h * im_w, 1)
    cam_pts_z.shape = (im_h * im_w, 1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:, :, 0]
    rgb_pts_g = color_img[:, :, 1]
    rgb_pts_b = color_img[:, :, 2]
    rgb_pts_r.shape = (im_h * im_w, 1)
    rgb_pts_g.shape = (im_h * im_w, 1)
    rgb_pts_b.shape = (im_h * im_w, 1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):
    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
                               (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(
        np.dot(cam_pose[0:3, 0:3], np.transpose(surface_pts)) + np.tile(cam_pose[0:3, 3:], (1, surface_pts.shape[0])))

    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:, 2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
        np.logical_and(surface_pts[:, 0] >= workspace_limits[0][0], surface_pts[:, 0] < workspace_limits[0][1]),
        surface_pts[:, 1] >= workspace_limits[1][0]), surface_pts[:, 1] < workspace_limits[1][1]),
        surface_pts[:, 2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    color_heightmap_r[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [0]]
    color_heightmap_g[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [1]]
    color_heightmap_b[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    # depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return color_heightmap, depth_heightmap


def mkdir(path, clean=False):
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def imretype(im, dtype):
    im = np.array(im)

    if im.dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(np.float)
    elif im.dtype == 'uint8':
        im = im.astype(np.float) / 255.
    elif im.dtype == 'uint16':
        im = im.astype(np.float) / 65535.
    else:
        raise NotImplementedError('unsupported source dtype: {0}'.format(im.dtype))

    assert np.min(im) >= 0 and np.max(im) <= 1

    if dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(dtype)
    elif dtype == 'uint8':
        im = (im * 255.).astype(dtype)
    elif dtype == 'uint16':
        im = (im * 65535.).astype(dtype)
    else:
        raise NotImplementedError('unsupported target dtype: {0}'.format(dtype))

    return im


def imwrite(path, obj):
    if not isinstance(obj, (collections.Sequence, collections.UserList)):
        obj = [obj]
    writer = imageio.get_writer(path)
    for im in obj:
        im = imretype(im, dtype='uint8').squeeze()
        if len(im.shape) == 3 and im.shape[0] == 3:
            im = np.transpose(im, (1, 2, 0))
        writer.append_data(im)
    writer.close()


def flow2im(flow, max=None, dtype='float32', cfirst=False):
    flow = np.array(flow)

    if np.ndim(flow) == 3 and flow.shape[0] == 2:
        x, y = flow[:, ...]
    elif np.ndim(flow) == 3 and flow.shape[-1] == 2:
        x, y = flow[..., :]
    else:
        raise NotImplementedError(
            'unsupported flow size: {0}'.format(flow.shape))

    rho, theta = cv2.cartToPolar(x, y)

    if max is None:
        max = np.maximum(np.max(rho), 1e-6)

    hsv = np.zeros(list(rho.shape) + [3], dtype=np.uint8)
    hsv[..., 0] = theta * 90 / np.pi
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(rho / max, 1) * 255

    im = cv2.cvtColor(hsv, code=cv2.COLOR_HSV2RGB)
    im = imretype(im, dtype=dtype)

    if cfirst:
        im = im.transpose(2, 0, 1)
    return im


def draw_arrow(image, action, direction_num=8, heightmap_size=[128, 128], heightmap_pixel_size=0.003):
    def put_in_bound(val, bound):
        # output: 0 <= val < bound
        val = min(max(0, val), bound - 1)
        return val

    img = image.copy()
    if isinstance(action, tuple):
        x_ini, y_ini, direction = action
    else:
        x_ini, y_ini, direction = action['2'], action['1'], action['0']

    arrow_scale = 0.003 / heightmap_pixel_size

    x_end = put_in_bound(int(x_ini + 15 * arrow_scale * np.cos(direction / direction_num * 2 * np.pi)), heightmap_size[1])
    y_end = put_in_bound(int(y_ini + 15 * arrow_scale * np.sin(direction / direction_num * 2 * np.pi)), heightmap_size[0])

    if img.shape[0] == 1:
        # gray img, white arrow
        img = imretype(img[0, :, :, np.newaxis], 'uint8')
        cv2.arrowedLine(img=img, pt1=(x_ini, y_ini), pt2=(x_end, y_end), color=255, tipLength=0.3)
    elif img.shape[2] == 3:
        # rgb img, red arrow
        cv2.arrowedLine(img=img, pt1=(x_ini, y_ini), pt2=(x_end, y_end), color=(255, 0, 0), tipLength=0.3)
    return img

def html_visualize(web_path:str, figures:dict, ids:list, cols:list, others:list=[], title:str='visualization'):
    '''
    others: list of dict
    'name': str, name of the data, visualize using h2()
    'data': string or ndarray(image)
    'width': int, width of the image (default 256)
    '''
    figure_path = osp.join(web_path, 'figures')
    mkdir(web_path, clean=True)
    mkdir(figure_path, clean=True)
    for figure_name, figure in figures.items():
        if not isinstance(figure, str):
            imwrite(osp.join(figure_path, figure_name + '.png'), figure)

    with dominate.document(title=title) as web:
        h1(title)
        with table(border=1, style='table-layout: fixed;'):
            with tr():
                with td(style='word-wrap: break-word;', halign='center', align='center', width='64px'):
                    p('id')
                for col in cols:
                    with td(style='word-wrap: break-word;', halign='center', align='center',):
                        p(col)
            for id in ids:
                with tr():
                    bgcolor = 'F1C073' if id.startswith('train') else 'C5F173'
                    with td(style='word-wrap: break-word;', halign='center', align='center', bgcolor=bgcolor):
                        for part in id.split('_'):
                            p(part)
                    for col in cols:
                        with td(style='word-wrap: break-word;', halign='center', align='top'):
                            s = figures['{}_{}'.format(id, col)]
                            if isinstance(s, str):
                                for x in s.split('\n'):
                                    p(x)
                            else:
                                img(style='width:128px', src=os.path.join('figures', '{}_{}.png'.format(id, col)))
        for idx, other in enumerate(others):
            h2(other['name'])
            if isinstance(other['data'], str):
                p(other['data'])
            else:
                imwrite(osp.join(figure_path, '_{}_{}.png'.format(idx, other['name'])), other['data'])
                img(style='width:{}px'.format(other.get('width', 256)), src=os.path.join('figures', '_{}_{}.png'.format(idx, other['name'])))
    with open(os.path.join(web_path, 'index.html'), 'w') as fp:
        fp.write(web.render())


def tsdf_loss(target, output):
    mask = torch.gt(target, torch.ones_like(target) * -0.99)
    loss = torch.mean((mask * (target - output))**2)

    return loss


# def get_mesh(tsdf_vol, voxel_size=0.006, vol_bnds=None, color_vol=None):
#     if vol_bnds is None:
#         vol_bnds = np.array([[0.308, 0.692], [-0.192, 0.192], [0.000, 0.096]])
#     S1, S2, S3 = tsdf_vol.shape
#     print(S1, S2, S3)
#     verts = []
#     colors = []
#     for x in range(S1):
#         for y in range(S2):
#             for z in range(S3):
#                 t = tsdf_vol[x][y][z]
#                 if t < 0:
#                     xx = vol_bnds[0][0] + voxel_size * x
#                     yy = vol_bnds[1][0] + voxel_size * y
#                     zz = vol_bnds[2][0] + voxel_size * z
#                     colors.append(-1 * t)
#                     verts.append([xx, yy, zz])
#     cmap = plt.get_cmap('jet')
#     colors = (cmap(np.asarray(colors))[...,:3] * 255).astype(np.uint8)
#     print(np.max(colors), np.min(colors))
    
    
#     vol_origin = vol_bnds[:,0].copy(order='C').astype(np.float32) # ensure C-order contigous
#     verts = np.asarray(verts)
#     faces = np.zeros([0,0])
#     norms = np.zeros([0,0])
#     colors = np.asarray(colors)
#     return verts,faces,norms,colors

# def meshwrite(filename,verts,faces,norms,colors):

#     # Write header
#     ply_file = open(filename,'w')
#     ply_file.write("ply\n")
#     ply_file.write("format ascii 1.0\n")
#     ply_file.write("element vertex %d\n"%(verts.shape[0]))
#     ply_file.write("property float x\n")
#     ply_file.write("property float y\n")
#     ply_file.write("property float z\n")
#     ply_file.write("property uchar red\n")
#     ply_file.write("property uchar green\n")
#     ply_file.write("property uchar blue\n")
#     ply_file.write("element face %d\n"%(faces.shape[0]))
#     ply_file.write("property list uchar int vertex_index\n")
#     ply_file.write("end_header\n")

#     # Write vertex list
#     for i in range(verts.shape[0]):
#         ply_file.write("%f %f %f %d %d %d\n"%(verts[i,0],verts[i,1],verts[i,2],colors[i,0],colors[i,1],colors[i,2]))

#     # Write face list
#     for i in range(faces.shape[0]):
#         ply_file.write("3 %d %d %d\n"%(faces[i,0],faces[i,1],faces[i,2]))

#     ply_file.close()

def get_mesh(tsdf_vol, voxel_size=0.006, vol_bnds=None, color_vol=None):
    if vol_bnds is None:
        vol_bnds = np.array([[0.308, 0.692], [-0.192, 0.192], [0.000, 0.096]])
    vol_origin = vol_bnds[:,0].copy(order='C').astype(np.float32) # ensure C-order contigous
        
    # Marching cubes
    verts,faces,norms,vals = measure.marching_cubes_lewiner(tsdf_vol,level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*voxel_size+vol_origin # voxel grid coordinates to world coordinates

    # Get vertex colors
    if color_vol is None:
        colors = np.ones_like(verts) * 255
        for i in range(verts.shape[0]):
            if verts[i][2] > 0.005:
                colors[i][0] = 0
    else:
        rgb_vals = color_vol[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]]
        colors_b = np.floor(rgb_vals/(256*256))
        colors_g = np.floor((rgb_vals-colors_b*256*256)/256)
        colors_r = rgb_vals-colors_b*256*256-colors_g*256
        colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
        colors = colors.astype(np.uint8)
    return verts,faces,norms,colors

def meshwrite(filename,verts,faces,norms,colors):

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n"%(faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(verts[i,0],verts[i,1],verts[i,2],norms[i,0],norms[i,1],norms[i,2],colors[i,0],colors[i,1],colors[i,2]))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n"%(faces[i,0],faces[i,1],faces[i,2]))

    ply_file.close()

# Save 3D mesh to a polygon .ply file
def objwrite(filename,verts,colors,faces=np.zeros([0, 4])):

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n"%(faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n"%(verts[i,0],verts[i,1],verts[i,2],colors[i,0],colors[i,1],colors[i,2]))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("4 %d %d %d %d\n"%(faces[i,0],faces[i,1],faces[i,2],faces[i,3]))

    ply_file.close()
