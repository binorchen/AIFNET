'''
    Author: Bin Chen, Lingyan Ruan
    Email: binorchen@gmail.com
    Date: Sep 7, 2021
'''

from tensorlayer.prepro import *
import os, scipy, numpy as np


def read_all_imgs(file_name_list):
    imgs = []
    for idx in range(0, len(file_name_list)):
        imgs.append(get_images(file_name_list[idx]))
    return imgs

def get_images(file_name):
    """ Input an image path and name, return an image array """
    image = (scipy.misc.imread(file_name, mode='RGB')/255.).astype(np.float32)

    new_shape_h = image.shape[0]-image.shape[0]%16 if image.shape[0]%16!=0 else image.shape[0]
    new_shape_w = image.shape[1]-image.shape[1]%16 if image.shape[1]%16!=0 else image.shape[1]

    image = image[0:new_shape_h, 0:new_shape_w]

    return image


    # Load the data files path
def load_test_data(img_path, gt_path, has_gt, dataset):
    df_img_files_name = list(os.path.join(img_path, name) for name in sorted(os.listdir(img_path)))
    gt_files_name = list(os.path.join(gt_path, name) for name in sorted(os.listdir(img_path))) if has_gt==True else []
    return df_img_files_name, gt_files_name


    # Load the data files path (use this when testing on all the LFDOF testing set)
def load_all_lfdof_testset(img_path, gt_path, has_gt, dataset):
    df_img_files_name,  gt_files_name = [], []
    fd_list = sorted(os.listdir(img_path))
    for fd in fd_list:
        last_path = os.path.join(img_path, fd)
        imgs = list(os.path.join(last_path, name) for name in os.listdir(last_path))
        df_img_files_name.extend(imgs)
        img_gt = os.path.join(gt_path, fd + '.png')
        gt_files_name.extend(img_gt for i in range(len(imgs)))

    return df_img_files_name, gt_files_name



