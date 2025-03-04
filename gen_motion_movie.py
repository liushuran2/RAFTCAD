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
from skimage import io

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

source_dir = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/NA_0.80_Hz_30_D_0_pow_150'  # 源文件夹
dirinfo = os.listdir(source_dir)
N_pair_per_file = 5  # 这个值需要根据你的实际情况设定
frame_N = 1000  # 这个值需要根据你的实际情况设定
scale_x = 2  # 根据你的实际情况设置
clip_length = frame_N // N_pair_per_file

# save path
save_path = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/test_dataset/scale_2'
raw_path_10mW = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/test_dataset/scale_2/10mW'
raw_path_30mW = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/test_dataset/scale_2/30mW'
raw_path_50mW = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/test_dataset/scale_2/50mW'
raw_path_70mW = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/test_dataset/scale_2/70mW'
raw_path_90mW = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/test_dataset/scale_2/90mW'
raw_path_110mW = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/test_dataset/scale_2/110mW'
raw_path_130mW = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/test_dataset/scale_2/130mW'
raw_path_150mW = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/test_dataset/scale_2/150mW'
gt_path = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/test_dataset/scale_2/gt'
# mkdir
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(raw_path_10mW):
    os.makedirs(raw_path_10mW)
if not os.path.exists(raw_path_30mW):
    os.makedirs(raw_path_30mW)
if not os.path.exists(raw_path_50mW):
    os.makedirs(raw_path_50mW)
if not os.path.exists(raw_path_70mW):
    os.makedirs(raw_path_70mW)
if not os.path.exists(raw_path_90mW):
    os.makedirs(raw_path_90mW)
if not os.path.exists(raw_path_110mW):
    os.makedirs(raw_path_110mW)
if not os.path.exists(raw_path_130mW):
    os.makedirs(raw_path_130mW)
if not os.path.exists(raw_path_150mW):
    os.makedirs(raw_path_150mW)

if not os.path.exists(gt_path):   
    os.makedirs(gt_path)

# imagePairs = np.zeros((len(dirinfo) * N_pair_per_file, 8, 512, 512), dtype=np.float32)  # 假设大小为 256x256
# motion_field = np.zeros((len(dirinfo) * N_pair_per_file, 8, 2, 512, 512), dtype=np.float32)

ind = 0

for i, dir_name in enumerate(dirinfo):
    curr_dir = os.path.join(source_dir, dir_name)
    
    # 读取两帧图像
    raw_file_name_10mW = os.path.join(curr_dir, 'Fsim_10mW.tiff')
    raw_frames_10mW = tiff.imread(raw_file_name_10mW)

    raw_file_name_30mW = os.path.join(curr_dir, 'Fsim_30mW.tiff')
    raw_frames_30mW = tiff.imread(raw_file_name_30mW)

    raw_file_name_50mW = os.path.join(curr_dir, 'Fsim_50mW.tiff')
    raw_frames_50mW = tiff.imread(raw_file_name_50mW)

    raw_file_name_70mW = os.path.join(curr_dir, 'Fsim_70mW.tiff')
    raw_frames_70mW = tiff.imread(raw_file_name_70mW)

    raw_file_name_90mW = os.path.join(curr_dir, 'Fsim_90mW.tiff')
    raw_frames_90mW = tiff.imread(raw_file_name_90mW)

    raw_file_name_110mW = os.path.join(curr_dir, 'Fsim_110mW.tiff')
    raw_frames_110mW = tiff.imread(raw_file_name_110mW)

    raw_file_name_130mW = os.path.join(curr_dir, 'Fsim_130mW.tiff')
    raw_frames_130mW = tiff.imread(raw_file_name_130mW)

    raw_file_name_150mW = os.path.join(curr_dir, 'Fsim_150mW.tiff')
    raw_frames_150mW = tiff.imread(raw_file_name_150mW)

    gt_file_name = os.path.join(curr_dir, 'Fsim_clean.tiff')
    gt_frames = tiff.imread(gt_file_name)
    

    for jjj in range(N_pair_per_file):
        raw_frames_out_10mW = raw_frames_10mW[jjj * clip_length: (jjj + 1) * clip_length]
        raw_frames_out_30mW = raw_frames_30mW[jjj * clip_length: (jjj + 1) * clip_length]
        raw_frames_out_50mW = raw_frames_50mW[jjj * clip_length: (jjj + 1) * clip_length]
        raw_frames_out_70mW = raw_frames_70mW[jjj * clip_length: (jjj + 1) * clip_length]
        raw_frames_out_90mW = raw_frames_90mW[jjj * clip_length: (jjj + 1) * clip_length]
        raw_frames_out_110mW = raw_frames_110mW[jjj * clip_length: (jjj + 1) * clip_length]
        raw_frames_out_130mW = raw_frames_130mW[jjj * clip_length: (jjj + 1) * clip_length]
        raw_frames_out_150mW = raw_frames_150mW[jjj * clip_length: (jjj + 1) * clip_length]
        gt_frames_out = gt_frames[jjj * clip_length: (jjj + 1) * clip_length]

        for frame_idx in range(1, clip_length):  # 从 2 到 8
            curr_frame_10mW = raw_frames_out_10mW[frame_idx]
            curr_frame_30mW = raw_frames_out_30mW[frame_idx]
            curr_frame_50mW = raw_frames_out_50mW[frame_idx]
            curr_frame_70mW = raw_frames_out_70mW[frame_idx]
            curr_frame_90mW = raw_frames_out_90mW[frame_idx]
            curr_frame_110mW = raw_frames_out_110mW[frame_idx]
            curr_frame_130mW = raw_frames_out_130mW[frame_idx]
            curr_frame_150mW = raw_frames_out_150mW[frame_idx]
            random_integer = random.randint(0, 4)
            u = np.random.rand(3 + random_integer, 3 + random_integer) - 0.5
            v = np.random.rand(3 + random_integer, 3 + random_integer) - 0.5
            u = u * scale_x
            v = v * scale_x

            # 调整 u 和 v 的大小
            u = zoom(u, (raw_frames_out_10mW.shape[1] / u.shape[0], raw_frames_out_10mW.shape[2] / u.shape[1]), order=3)
            v = zoom(v, (raw_frames_out_10mW.shape[1] / v.shape[0], raw_frames_out_10mW.shape[2] / v.shape[1]), order=3)

            # 进行图像扭曲
            # warp_frame_2 = imregister_wrapper(frame_2, u, v)
            w = np.stack((u, v), axis=-1)
            warp_frame_10mW = image_warp(curr_frame_10mW, w)
            warp_frame_30mW = image_warp(curr_frame_30mW, w)
            warp_frame_50mW = image_warp(curr_frame_50mW, w)
            warp_frame_70mW = image_warp(curr_frame_70mW, w)
            warp_frame_90mW = image_warp(curr_frame_90mW, w)
            warp_frame_110mW = image_warp(curr_frame_110mW, w)
            warp_frame_130mW = image_warp(curr_frame_130mW, w)
            warp_frame_150mW = image_warp(curr_frame_150mW, w)

            # output frame_out
            raw_frames_out_10mW[frame_idx] = warp_frame_10mW
            raw_frames_out_30mW[frame_idx] = warp_frame_30mW
            raw_frames_out_50mW[frame_idx] = warp_frame_50mW
            raw_frames_out_70mW[frame_idx] = warp_frame_70mW
            raw_frames_out_90mW[frame_idx] = warp_frame_90mW
            raw_frames_out_110mW[frame_idx] = warp_frame_110mW
            raw_frames_out_130mW[frame_idx] = warp_frame_130mW
            raw_frames_out_150mW[frame_idx] = warp_frame_150mW
        
        # save raw_frames_out as tiff
        raw_frames_out_10mW = raw_frames_out_10mW.astype(np.uint16)
        raw_frames_out_10mW = np.clip(raw_frames_out_10mW, 0, 65535)
        io.imsave(os.path.join(raw_path_10mW, f'{str(ind+1).rjust(3, "0")}.tiff'), raw_frames_out_10mW)

        raw_frames_out_30mW = raw_frames_out_30mW.astype(np.uint16)
        raw_frames_out_30mW = np.clip(raw_frames_out_30mW, 0, 65535)
        io.imsave(os.path.join(raw_path_30mW, f'{str(ind+1).rjust(3, "0")}.tiff'), raw_frames_out_30mW)

        raw_frames_out_50mW = raw_frames_out_50mW.astype(np.uint16)
        raw_frames_out_50mW = np.clip(raw_frames_out_50mW, 0, 65535)
        io.imsave(os.path.join(raw_path_50mW, f'{str(ind+1).rjust(3, "0")}.tiff'), raw_frames_out_50mW)

        raw_frames_out_70mW = raw_frames_out_70mW.astype(np.uint16)
        raw_frames_out_70mW = np.clip(raw_frames_out_70mW, 0, 65535)
        io.imsave(os.path.join(raw_path_70mW, f'{str(ind+1).rjust(3, "0")}.tiff'), raw_frames_out_70mW)

        raw_frames_out_90mW = raw_frames_out_90mW.astype(np.uint16)
        raw_frames_out_90mW = np.clip(raw_frames_out_90mW, 0, 65535)
        io.imsave(os.path.join(raw_path_90mW, f'{str(ind+1).rjust(3, "0")}.tiff'), raw_frames_out_90mW)

        raw_frames_out_110mW = raw_frames_out_110mW.astype(np.uint16)
        raw_frames_out_110mW = np.clip(raw_frames_out_110mW, 0, 65535)
        io.imsave(os.path.join(raw_path_110mW, f'{str(ind+1).rjust(3, "0")}.tiff'), raw_frames_out_110mW)

        raw_frames_out_130mW = raw_frames_out_130mW.astype(np.uint16)
        raw_frames_out_130mW = np.clip(raw_frames_out_130mW, 0, 65535)
        io.imsave(os.path.join(raw_path_130mW, f'{str(ind+1).rjust(3, "0")}.tiff'), raw_frames_out_130mW)

        raw_frames_out_150mW = raw_frames_out_150mW.astype(np.uint16)
        raw_frames_out_150mW = np.clip(raw_frames_out_150mW, 0, 65535)
        io.imsave(os.path.join(raw_path_150mW, f'{str(ind+1).rjust(3, "0")}.tiff'), raw_frames_out_150mW)

        gt_frames_out = gt_frames_out.astype(np.uint16)
        gt_frames_out = np.clip(gt_frames_out, 0, 65535)
        io.imsave(os.path.join(gt_path, f'{str(ind+1).rjust(3, "0")}.tiff'), gt_frames_out)

        ind += 1
        print(ind)



