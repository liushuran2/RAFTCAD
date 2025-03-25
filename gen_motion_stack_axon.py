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

source_dir = '/mnt/nas01/LSR/DATA/2p_bench/HP01/HCA301-Axon-P5-oir/HCA301-Axon-P5-oir'  # 源文件夹
dirinfo = os.listdir(source_dir)
N_pair_per_file = 20  # 这个值需要根据你的实际情况设定
frame_N = 200  # 这个值需要根据你的实际情况设定
scale_x = 10  # 根据你的实际情况设置
stack_length = 24  # 根据你的实际情况设置

imagePairs = np.zeros((9 * N_pair_per_file, stack_length, 512, 512), dtype=np.float32)  # 假设大小为 256x256

ind = 0

file_name_list = []
for file in dirinfo:
    if file.endswith('-1.tif'):
        file_name_list.append(os.path.join(source_dir, file))
# file_name_list.append(source_dir + 'HCA301-250109-ABC-2-P5-AXON_00001-1.tif')
# file_name_list.append(source_dir + '97288_20210315_00003.tif')
# file_name_list.append('Fsim_30mW.tiff')
# file_name_list.append('Fsim_30mW.tiff')
for i, file_name in enumerate(file_name_list):
    # curr_dir = os.path.join(source_dir, dir_name)
    
    # 读取两帧图像
    # file_name = os.path.join(curr_dir, 'Fsim_30mW.tiff')
    # neuron_mask_path = os.path.join(curr_dir, 'NeuronMask.mat')  # 假设此处是读取 `.mat` 文件
    frames = tiff.imread(file_name)

    for jjj in range(N_pair_per_file):
        # 获取两帧
        index_1 = random.randint(0, frame_N - 50)
        frame_1 = frames[index_1]
        
        imagePairs[ind, 0, :, :] = frame_1.astype(np.float32)

        for frame_idx in range(2, stack_length + 1):  # 从 2 到 24
            index_2 = index_1 + frame_idx - 1
            frame_2 = frames[index_2]

            warp_frame_2 = frame_2
            imagePairs[ind, frame_idx - 1, :, :] = warp_frame_2.astype(np.float32)

        max_val = np.max(imagePairs[ind, :, :, :])
        # imagePairs[ind, :, :, :] /= max_val  # 如果需要归一化

        ind += 1
        print(ind)


# 定义文件名
filename = f'/mnt/nas02/LSR/DATA/NAOMi_dataset/N_{imagePairs.shape[0]}_axon_stack_24.h5'

# 创建 HDF5 文件并保存数据
with h5py.File(filename, 'w') as f:
    # 创建并写入 '/image_pairs' 数据集
    f.create_dataset('/image_pairs', data=imagePairs, dtype='float32')


