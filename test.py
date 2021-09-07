'''
    Author: Bin Chen, Lingyan Ruan
    Email: binorchen@gmail.com
    Date: Sep 7, 2021
'''

import os, time, numpy as np
from utils import *
from model import *

import tensorflow as tf
import tensorlayer as tl

from skimage import measure


def evaluate(dataset, row, col, path, gt_path, has_gt, w_name, out_path):
    print('Evaluation Start...')

    # create necessary folders
    save_dir = os.path.join(out_path + dataset)
    out_dfm_path = os.path.join(save_dir, 'out_dfm')
    out_dfm_norm_path = os.path.join(save_dir, 'out_dfm_norm')
    out_aif_path = os.path.join(save_dir, 'out_aif')
    tl.files.exists_or_mkdir(out_dfm_path, verbose = True)
    tl.files.exists_or_mkdir(out_dfm_norm_path, verbose = True)
    tl.files.exists_or_mkdir(out_aif_path, verbose = True)
    
    if has_gt==True:
        # open txt files to save PSNR and SSIM values
        f_psnr = open(os.path.join(save_dir, 'psnr.txt'), 'w+')
        f_ssim = open(os.path.join(save_dir, 'ssim.txt'), 'w+')

    test_h = row - row%16 if row%16 != 0 else row
    test_w = col - col%16 if col%16 != 0 else col

    # load testing images
    if dataset == 'LFDOF' or dataset == 'DPD' or dataset == "RTF":
        test_df_img_list, test_gt_img_list = load_test_data(path, gt_path, has_gt, dataset)
    else:
        test_df_img_list = [path]
        test_gt_img_list = []

    # test_df_img_list, test_gt_img_list = load_all_lfdof_testset(path, gt_path, has_gt, dataset) # use this when testing on all the LFDOF testing set, in which multiple one ground truth correspond to multiple inputs
    
    # define testing session
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))

    # == Define Model ========================
    # define input
    with tf.variable_scope('input'):
        patches_df_lf = tf.placeholder('float32', [1, test_h, test_w, 3], name = 'input_df_lf')
        labels_lf_aif = tf.placeholder('float32', [1, test_h, test_w, 3], name = 'labels_lf_aif')
    # define defocus network
    with tf.variable_scope('defocus_net') as scope:
        with tf.variable_scope('encoder') as scope:
            feats_lf_down = VGG19_down(patches_df_lf, reuse = False, scope = scope, is_test = True)
        with tf.variable_scope('decoder') as scope:
            output_lf_defocus, _, _, _ = UNet_up(patches_df_lf, feats_lf_down, is_train = False, reuse = False, scope = scope)
    # define deblur network
    with tf.variable_scope('deblur_net') as scope:
            lf_ds_0_aif = deblur_net(output_lf_defocus, patches_df_lf, is_train = False, reuse = False, scope = scope)

    # == Read Test Data ======================
    avg_time, avg_psnr, avg_ssim = 0., 0., 0.

    # initialize variables
    sess.run(tf.global_variables_initializer())
    # load pre-trained weights file
    tl.files.load_and_assign_npz_dict(name = w_name, sess = sess)

    # == Read Test Data ======================
    for i in np.arange(len(test_df_img_list)):
        df_img_list = [test_df_img_list[i]]
        test_df_imgs = read_all_imgs(df_img_list)
        df_img = np.expand_dims(test_df_imgs[0], 0)
        if has_gt==True:    
            aif_img_list = [test_gt_img_list[i]]
            test_aif_imgs = read_all_imgs(aif_img_list)
            aif_img = np.expand_dims(test_aif_imgs[0], 0) 

        print('processing {} ...'.format(test_df_img_list[i]))

        tic = time.time()
        # feed data
        feed_dict = {patches_df_lf: df_img, labels_lf_aif:aif_img} if has_gt==True else {patches_df_lf: df_img}
        # -- run the testing ----------------
        est_dfm, est_aif = sess.run([output_lf_defocus, lf_ds_0_aif.outputs], feed_dict)
        toc = time.time()

        print('processing {} ... Done [{:.3f}s]'.format(test_df_img_list[i], toc - tic))
        avg_time = avg_time + (toc - tic)

        est_dfm = np.squeeze(est_dfm)
        dfm_temp = est_dfm - est_dfm.min()
        dfm_norm = dfm_temp / dfm_temp.max()

        img_name = test_df_img_list[i].split('/')[-1]

        scipy.misc.toimage(est_dfm, cmin = 0., cmax = 1.).save(out_dfm_path + '/' + img_name)
        scipy.misc.toimage(dfm_norm, cmin = 0., cmax = 1.).save(out_dfm_norm_path + '/' + img_name)
        scipy.misc.toimage(np.squeeze(est_aif), cmin = 0., cmax = 1.).save(out_aif_path + '/' + img_name)

        if has_gt==True:
            img_psnr = measure.compare_psnr(np.squeeze(aif_img), np.squeeze(est_aif))
            img_ssim = measure.compare_ssim(np.squeeze(aif_img), np.squeeze(est_aif), multichannel=True, data_range=1.0)
            print('[PSNR and SSIM]: ', img_psnr, ' ', img_ssim)
            avg_psnr = avg_psnr + img_psnr
            avg_ssim = avg_ssim + img_ssim
            f_psnr.write(str(img_psnr)+'\n')
            f_ssim.write(str(img_ssim)+'\n')

    avg_time = avg_time / len(test_df_img_list)
    print('averge processing time: {:.3f}s'.format(avg_time))

    if has_gt==True:
        avg_psnr = avg_psnr / len(test_df_img_list)
        avg_ssim = avg_ssim / len(test_df_img_list)
        print('averge PSNR: {:.3f}'.format(avg_psnr))
        print('averge SSIM: {:.3f}'.format(avg_ssim))
        f_psnr.close()
        f_ssim.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network', type = str , default = 'AIFNet', help = 'which network')
    parser.add_argument('-d', '--dataset', type = str , default = 'LFDOF', help = 'which evaluation dataset')
    parser.add_argument('-r', '--row', type = int , default = 688, help = 'image size (row)')
    parser.add_argument('-c', '--col', type = int , default = 1008, help = 'image size (column)')
    parser.add_argument('-p', '--path', type = str , default = './test_set/LFDOF/input/', help = 'the path of testing images')
    parser.add_argument('-gt', '--has_gt', type = str , default = 1, help = 'ground truth is available or not')
    parser.add_argument('-gtp', '--gt_path', type = str , default = './test_set/LFDOF/ground_truth/', help = 'the path of the ground truth of testing images')
    parser.add_argument('-w', '--w_name', type = str , default = './weights/aifnet_pretrained.npz', help = 'name of weights file')
    parser.add_argument('-op', '--out_path', type = str , default = './output/', help = 'the path to save testing results')

    args = parser.parse_args()

    evaluate(dataset = args.dataset, 
            row = args.row, 
            col = args.col, 
            path = args.path, 
            gt_path = args.gt_path, 
            has_gt = args.has_gt, 
            w_name = args.w_name,
            out_path = args.out_path)