import torch
import numpy as np
import sys
import argparse
import errno, os
import glob
import os.path as osp
import re
# import matplotlib.pyplot as plt
import cv2
from PIL import Image
import gc
import open3d as o3d
import matplotlib.pyplot as plt
import json
from warp_func import *

parser = argparse.ArgumentParser(description='Depth fusion with consistency check.')
parser.add_argument('--root_path', type=str, default='/Users/shuocheng/Projects/scan_data/'
                                                     'new_calib_data_13-02-2024-21-30-38')
parser.add_argument('--save_path', type=str, default='./points')

parser.add_argument('--dist_thresh', type=float, default=0.0005)

parser.add_argument('--num_consist', type=int, default=15)
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)
    file.close()

def write_ply(file, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    o3d.io.write_point_cloud(file, pcd, write_ascii=False)

def filter_depth(ref_depth, src_depths, ref_proj, src_projs):
    '''
	
	:param ref_depth: (1, 1, H, W)
	:param src_depths: (B, 1, H, W)
	:param ref_proj: (1, 4, 4)
	:param src_proj: (B, 4, 4)
	:return: ref_pc: (1, 3, H, W), aligned_pcs: (B, 3, H, W), dist: (B, 1, H, W)
	'''

    ref_pc = generate_points_from_depth(ref_depth, ref_proj)
    src_pcs = generate_points_from_depth(src_depths, src_projs)

    aligned_pcs = homo_warping(src_pcs, src_projs, ref_proj, ref_depth)

    # _, axs = plt.subplots(3, 4)
    # for i in range(3):
    # 	axs[i, 0].imshow(src_pcs[0, i], vmin=0, vmax=1)
    # 	axs[i, 1].imshow(aligned_pcs[0, i],  vmin=0, vmax=1)
    # 	axs[i, 2].imshow(ref_pc[0, i],  vmin=0, vmax=1)
    # 	axs[i, 3].imshow(ref_pc[0, i] - aligned_pcs[0, i], vmin=-0.5, vmax=0.5, cmap='coolwarm')
    # plt.show()

    x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0]) ** 2
    y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1]) ** 2
    z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2]) ** 2
    dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1)

    return ref_pc, aligned_pcs, dist


calib_data = {
     "2-14": np.array([[-0.00608726, -0.99968002, -0.02455219, 0.04766486],
                                [0.99838566, -0.00746227, 0.05630625, -0.0463541],
                                [-0.05647145, -0.02416981, 0.99811161, -0.07277737],
                                [0., 0., 0., 1.]]),
              "2-13": np.array([[-0.00473627, -0.99850071, -0.0545334 ,  0.04572007],
       [ 0.99997214, -0.00441448, -0.00601973, -0.03668707],
       [ 0.00576997, -0.05456039,  0.9984938 , -0.10900777],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
              "2-13_03": np.array([[-0.00473627, -0.99850071, -0.0545334 ,  0.04572007],
       [ 0.99997214, -0.00441448, -0.00601973, -0.03668707],
       [ 0.00576997, -0.05456039,  0.9984938 , -0.03900777],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
              "2-13_00": np.array([[-0.00473627, -0.99850071, -0.0545334 ,  0.04572007],
       [ 0.99997214, -0.00441448, -0.00601973, -0.03668707],
       [ 0.00576997, -0.05456039,  0.9984938 , -0.000010900777],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
}

def load_data2(root_path, calib_tag):
    def load_json(file_path):
        with open(file_path) as json_file:
            json_data = json.load(json_file)
            return json_data

    rgb_paths = glob.glob(f"{root_path}/obj_imgs/*.png")
    rgb_paths = sorted(rgb_paths)
    json_data = load_json(f"{root_path}/0_ee_poses_scan_obj.json")
    # cam2ee = np.array(json_data["cam2ee"])

    ee_poses = np.array(json_data["eef_poses"])
    assert len(ee_poses) == len(rgb_paths)
    rgbs = []
    depths = []
    projs = []

    intr_mat = np.array([[606.12524414, 0., 318.55557251],
                  [0., 605.62731934, 250.66503906],
                  [0., 0., 1.]])

    cam2ee = calib_data[calib_tag]

    for i, rgb_path in enumerate(rgb_paths):
        rgb = np.array(Image.open(rgb_path), dtype="float32")
        rgbs.append(rgb)

        dep = np.load(f"{root_path}/obj_deps/{i:06d}.npy", allow_pickle=True)
        dep = np.array(dep) / 1000.
        msk = np.array(Image.open(f"{root_path}/obj_msks/{i:06d}.png"))
        dep *= msk

        depths.append(torch.from_numpy(dep).unsqueeze(0))

        ee_pose = ee_poses[i]
        cam2base = ee_pose @ cam2ee
        extr_mat = np.linalg.inv(cam2base)

        proj_mat = np.eye(4)
        proj_mat[:3, :4] = np.dot(intr_mat[:3, :3], extr_mat[:3, :4])
        projs.append(torch.from_numpy(proj_mat))

    depths = torch.stack(depths).float()
    projs = torch.stack(projs).float()
    return depths, projs, rgbs


def extract_points(pc, mask, rgb):
    pc = pc.cpu().numpy()
    mask = mask.cpu().numpy()

    mask = np.reshape(mask, (-1,))
    pc = np.reshape(pc, (-1, 3))
    rgb = np.reshape(rgb, (-1, 3))

    points = pc[np.where(mask)]
    colors = rgb[np.where(mask)]

    points_with_color = np.concatenate([points, colors], axis=1)

    return points_with_color


def main(tag_name):
    mkdir_p(args.save_path)
    scene = "scan_obj"

    depths, projs, rgbs = load_data2(args.root_path, calib_tag=tag_name)
    tot_frame = depths.shape[0]
    height, width = depths.shape[2], depths.shape[3]
    points = []

    print('Scene: {} total: {} frames'.format(scene, tot_frame))
    for i in range(tot_frame):
        pc_buff = torch.zeros((3, height, width), device=depths.device, dtype=depths.dtype)
        val_cnt = torch.zeros((1, height, width), device=depths.device, dtype=depths.dtype)
        j = 0
        batch_size = 20

        while True:
            ref_pc, pcs, dist = filter_depth(ref_depth=depths[i:i + 1],
                                             src_depths=depths[j:min(j + batch_size, tot_frame)],
                                             ref_proj=projs[i:i + 1],
                                             src_projs=projs[j:min(j + batch_size, tot_frame)])
            masks = (dist < args.dist_thresh).float()
            masked_pc = pcs * masks
            pc_buff += masked_pc.sum(dim=0, keepdim=False)
            val_cnt += masks.sum(dim=0, keepdim=False)

            j += batch_size
            if j >= tot_frame:
                break

        final_mask = (val_cnt >= args.num_consist).squeeze(0)
        avg_points = torch.div(pc_buff, val_cnt).permute(1, 2, 0)

        final_pc = extract_points(avg_points, final_mask, rgbs[i])
        points.append(final_pc)
        print('Processing {} {}/{} ...'.format(scene, i + 1, tot_frame))
    # np.save('{}/{}/{:08d}.npy'.format(args.save_path, scene, i+1), final_pc)

    del depths, rgbs, projs
    gc.collect()

    full_points = np.concatenate(points, axis=0)
    inds = list(range(len(full_points)))
    np.random.shuffle(inds)
    full_points = full_points[inds]
    full_points = full_points[:4096*2]
    write_ply('{}/{}_{}.ply'.format(args.save_path, scene, tag_name), full_points)
    del points

    gc.collect()
    print('Save {}/.ply successful.'.format(args.save_path, ))



if __name__ == '__main__':
    with torch.no_grad():
        for key in calib_data:
            main(tag_name=key)
