import os
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import random
import h5py
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.transform import warp, AffineTransform
import tifffile as tiff
from utils.frame_utils import image_warp

# def imregister_wrapper(f2, u, v, f1=None):
#     if f1 is None:
#         f1 = f2
    
#     # Combine u and v into a 2-channel displacement field
#     w = np.zeros((*u.shape, 2), dtype=np.float32)
#     w[..., 0] = u
#     w[..., 1] = v
    
#     # Call the wrapper function
#     return imregister_wrapper_w(f2, w, f1)

# def imregister_wrapper_w(f2, w, f1, interpolation_method='cubic'):
#     if f1 is None:
#         f1 = f2
    
#     registered = np.zeros_like(f2, dtype=np.float32)
    
#     # Perform the warp operation for each channel (if f2 is a multi-channel image)
#     registered = imwarp(f2, w, interpolation_method)
    
#     # Handle NaN values in the warped image and replace them with the corresponding values from f1
#     registered[np.isnan(registered)] = f1[np.isnan(registered)]
    
#     return registered

# def imwarp(image, flow_field, interpolation_method='cubic'):
#     """
#     This function warps the image using the displacement field `flow_field`.
#     It supports cubic interpolation.
#     """
#     assert image.shape[:2] == flow_field.shape[:2], \
#         f"Image and flow field dimensions must match: {image.shape}, {flow_field.shape}"

#     # Create meshgrid for coordinates
#     grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

#     # Compute displaced coordinates
#     coords_x = grid_x + flow_field[..., 0]
#     coords_y = grid_y + flow_field[..., 1]

#     # Perform interpolation using map_coordinates
#     if interpolation_method == 'cubic':
#         warped_image = map_coordinates(image, [coords_y, coords_x], order=3, mode='nearest', cval=np.nan)
#     else:
#         # Default to nearest neighbor if interpolation method is unknown
#         warped_image = map_coordinates(image, [coords_y, coords_x], order=1, mode='nearest', cval=np.nan)

#     return warped_image


# from your_module import imregister_wrapper

source_dir = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200/NA_0.80_Hz_30_D_0_pow_150'  # 源文件夹
dirinfo = os.listdir(source_dir)
N_pair_per_file = 70  # 这个值需要根据你的实际情况设定
frame_N = 1000  # 这个值需要根据你的实际情况设定
scale_x = 10  # 根据你的实际情况设置

imagePairs = np.zeros((len(dirinfo) * N_pair_per_file, 8, 512, 512), dtype=np.float32)  # 假设大小为 256x256
motion_field = np.zeros((len(dirinfo) * N_pair_per_file, 8, 2, 512, 512), dtype=np.float32)

ind = 0

for i, dir_name in enumerate(dirinfo):
    curr_dir = os.path.join(source_dir, dir_name)
    
    # 读取两帧图像
    file_name = os.path.join(curr_dir, 'Fsim_30mW.tiff')
    # neuron_mask_path = os.path.join(curr_dir, 'NeuronMask.mat')  # 假设此处是读取 `.mat` 文件
    frames = tiff.imread(file_name)

    for jjj in range(N_pair_per_file):
        # 获取两帧
        index_1 = random.randint(0, frame_N - 10)
        frame_1 = frames[index_1]
        
        imagePairs[ind, 0, :, :] = frame_1.astype(np.float32)

        for frame_idx in range(2, 9):  # 从 2 到 8
            index_2 = index_1 + frame_idx - 1
            frame_2 = frames[index_2]

            random_integer = random.randint(0, 10)
            u = np.random.rand(5 + random_integer, 5 + random_integer) - 0.5
            v = np.random.rand(5 + random_integer, 5 + random_integer) - 0.5
            u = u * scale_x
            v = v * scale_x

            # 调整 u 和 v 的大小
            u = zoom(u, (frame_1.shape[0] / u.shape[0], frame_1.shape[1] / u.shape[1]), order=3)
            v = zoom(v, (frame_1.shape[0] / v.shape[0], frame_1.shape[1] / v.shape[1]), order=3)

            # 进行图像扭曲
            # warp_frame_2 = imregister_wrapper(frame_2, u, v)
            w = np.stack((u, v), axis=-1)
            warp_frame_2 = image_warp(frame_2, w)
            imagePairs[ind, frame_idx - 1, :, :] = warp_frame_2.astype(np.float32)
            motion_field[ind, frame_idx - 1, 0, :, :] = u.astype(np.float32)
            motion_field[ind, frame_idx - 1, 1, :, :] = v.astype(np.float32)

        max_val = np.max(imagePairs[ind, :, :, :])
        # imagePairs[ind, :, :, :] /= max_val  # 如果需要归一化

        ind += 1
        print(ind)


# 定义文件名
filename = f'/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200/N_{imagePairs.shape[0]}_scale_{scale_x}_stack_8_multiscale_30mW.h5'

# 创建 HDF5 文件并保存数据
with h5py.File(filename, 'w') as f:
    # 创建并写入 '/image_pairs' 数据集
    f.create_dataset('/image_pairs', data=imagePairs, dtype='float32')

    # 创建并写入 '/motions' 数据集
    f.create_dataset('/motions', data=motion_field, dtype='float32')


