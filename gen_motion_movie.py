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
scale_x = 10  # 根据你的实际情况设置
clip_length = frame_N // N_pair_per_file

# save path
save_path = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/10mW'
raw_path = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/10mW/raw'
gt_path = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/10mW/gt'
# mkdir
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(raw_path):
    os.makedirs(raw_path)
if not os.path.exists(gt_path):   
    os.makedirs(gt_path)

# imagePairs = np.zeros((len(dirinfo) * N_pair_per_file, 8, 512, 512), dtype=np.float32)  # 假设大小为 256x256
# motion_field = np.zeros((len(dirinfo) * N_pair_per_file, 8, 2, 512, 512), dtype=np.float32)

ind = 0

for i, dir_name in enumerate(dirinfo):
    curr_dir = os.path.join(source_dir, dir_name)
    
    # 读取两帧图像
    raw_file_name = os.path.join(curr_dir, 'Fsim_10mW.tiff')
    raw_frames = tiff.imread(raw_file_name)

    gt_file_name = os.path.join(curr_dir, 'Fsim_clean.tiff')
    gt_frames = tiff.imread(gt_file_name)
    

    for jjj in range(N_pair_per_file):
        raw_frames_out = raw_frames[jjj * clip_length: (jjj + 1) * clip_length]
        gt_frames_out = gt_frames[jjj * clip_length: (jjj + 1) * clip_length]

        for frame_idx in range(1, clip_length):  # 从 2 到 8
            curr_frame = raw_frames_out[frame_idx]
            random_integer = random.randint(0, 3)
            u = np.random.rand(5 + random_integer, 5 + random_integer) - 0.5
            v = np.random.rand(5 + random_integer, 5 + random_integer) - 0.5
            u = u * scale_x
            v = v * scale_x

            # 调整 u 和 v 的大小
            u = zoom(u, (raw_frames_out.shape[1] / u.shape[0], raw_frames_out.shape[2] / u.shape[1]), order=3)
            v = zoom(v, (raw_frames_out.shape[1] / v.shape[0], raw_frames_out.shape[2] / v.shape[1]), order=3)

            # 进行图像扭曲
            # warp_frame_2 = imregister_wrapper(frame_2, u, v)
            w = np.stack((u, v), axis=-1)
            warp_frame_2 = image_warp(curr_frame, w)

            # output frame_out
            raw_frames_out[frame_idx] = warp_frame_2
        
        # save raw_frames_out as tiff
        raw_frames_out = raw_frames_out.astype(np.uint16)
        raw_frames_out = np.clip(raw_frames_out, 0, 65535)
        io.imsave(os.path.join(raw_path, f'{ind+1}.tiff'), raw_frames_out)
        gt_frames_out = gt_frames_out.astype(np.uint16)
        gt_frames_out = np.clip(gt_frames_out, 0, 65535)
        io.imsave(os.path.join(gt_path, f'{ind+1}.tiff'), gt_frames_out)



        ind += 1
        print(ind)



