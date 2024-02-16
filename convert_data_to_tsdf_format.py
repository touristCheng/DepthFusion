import numpy as np
import cv2
import os
import json

def load_json(file_path):
    with open(file_path) as json_file:
        json_data = json.load(json_file)
        return json_data

root_dir = "//mnt/data2/shuo/rss24_rebuttal/new_calib_data_13-02-2024-21-30-38"

npy_folder = f'{root_dir}/obj_deps/'
tsdf_folder = f'{root_dir}/tsdf_data/'
json_data = load_json(f'{root_dir}/0_ee_poses_scan_obj.json')


all_npy_files = os.listdir(npy_folder)
all_npy_files = sorted(all_npy_files)

ee_poses = np.array(json_data["eef_poses"])
assert len(ee_poses) == len(all_npy_files)

cam2ee = np.array([[-0.00608726, -0.99968002, -0.02455219, 0.04766486],
                   [0.99838566, -0.00746227, 0.05630625, -0.0463541],
                   [-0.05647145, -0.02416981, 0.99811161, -0.07277737],
                   [0., 0., 0., 1.]])

for itr, npy_file in enumerate(all_npy_files):
    # read depth from npy file and save it as png
    depth = np.load(npy_folder + npy_file)

    save_dep_path = tsdf_folder + "frame-" + npy_file.replace('.npy', '.depth.png')
    os.makedirs(os.path.dirname(save_dep_path), exist_ok=True)
    cv2.imwrite(save_dep_path, depth)

    eef_pose = ee_poses[itr]
    cam2world = eef_pose @ cam2ee

    save_txt_path = tsdf_folder + "frame-" + npy_file.replace('.npy', '.pose.txt')
    os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
    np.savetxt(save_txt_path, cam2world, fmt='%1.7e', delimiter='\t')

rgb_folder = f'{root_dir}/obj_imgs/'
all_png_files = os.listdir(rgb_folder)
all_png_files = sorted(all_png_files)

for png_file in all_png_files:
    image = cv2.imread(rgb_folder + png_file, cv2.IMREAD_UNCHANGED)
    save_rgb_path = tsdf_folder + "frame-" + png_file.replace('.png', '.color.png')
    os.makedirs(os.path.dirname(save_rgb_path), exist_ok=True)
    cv2.imwrite(save_rgb_path, image)

intrinsic_K = np.array(
[[606.12524414,   0.,         318.55557251],
 [  0. ,        605.62731934, 250.66503906],
 [  0. ,          0.  ,         1.        ]]
)

np.savetxt(tsdf_folder + 'camera-intrinsics.txt', intrinsic_K, fmt='%1.7e', delimiter='\t')