'''
    Part of this code is adapted from https://github.com/codeslake/DMENet
    Author: Bin Chen, Lingyan Ruan
    Email: binorchen@gmail.com
    Date: Sep 7, 2021
'''

import tensorflow as tf
import numpy as np
from tensorlayer.layers import *

max_coc = 29

def VGG19_down(rgb, reuse, scope, is_test = False):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope(scope, reuse = reuse):
        rgb_scaled = rgb * 255.0
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)

        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv1_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_2')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv1_2')
        d0 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv2_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_2')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv2_2')
        d1 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv3_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv3_2')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv3_3')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_4')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv3_4')
        d2 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv4_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv4_2')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv4_3')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_4')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv4_4')
        d3 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')                           
        """ conv5 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv5_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv5_2')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv5_3')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_4')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv5_4')
        d4 = network

        if is_test == False:
            logits = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad6_1')
            logits = Conv2d(logits, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv6_1')

            logits = logits.outputs
            size = logits.get_shape().as_list()
            logits = InputLayer(logits)
            logits = Conv2d(logits, n_filter=512, filter_size=(size[1], size[2]), strides=(1, 1), act=tf.nn.relu, padding='VALID', name='c_logits_1')
            logits = FlattenLayer(logits, name='flatten')
            logits = DenseLayer(logits, n_units=512, act=tf.nn.relu, W_init = w_init_relu, name='c_logits_1')
            logits = DenseLayer(logits, n_units=1, act=tf.identity, W_init = w_init_sigmoid, name='c_logits_2')
            
            return network, [d0.outputs, d1.outputs, d2.outputs, d3.outputs, d4.outputs], d3.outputs, logits.outputs
        else:

            return [d0.outputs, d1.outputs, d2.outputs, d3.outputs, d4.outputs]

def UNet_up(images, feats, is_train=False, reuse=False, scope = 'unet_up'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    g_init = None
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    def UpSampling2dLayer_(input, scale, method, align_corners, name):
        input = input.outputs
        size = tf.shape(input)

        n = InputLayer(input, name = name + '_in')
        n = UpSampling2dLayer(n, size=[size[1] * scale[0], size[2] * scale[1]], is_scale = False, method = method, align_corners = align_corners, name = name)

        return n

    with tf.variable_scope(scope, reuse=reuse):
        d0 = InputLayer(feats[0], name='d0')
        d1 = InputLayer(feats[1], name='d1')
        d2 = InputLayer(feats[2], name='d2')
        d3 = InputLayer(feats[3], name='d3')
        d4 = InputLayer(feats[4], name='d4')

        u4 = d4
        u4 = PadLayer(u4, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4_aux/pad1')
        u4 = Conv2d(u4, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u4_aux/c1')
        u4 = BatchNormLayer(u4, act=lrelu, is_train = is_train, gamma_init = g_init, name='u4_aux/b1')
        u4 = PadLayer(u4, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4_aux/pad2')
        u4 = Conv2d(u4, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u4_aux/c2')
        u4 = BatchNormLayer(u4, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u4_aux/b2')
        u4 = u4.outputs

        n = UpSampling2dLayer_(d4, (2, 2), method = 1, align_corners=True, name='u3/u')
        n = ConcatLayer([n, d3], concat_dim = 3, name='u3/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad3')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b3')

        u3 = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3_aux/pad1')
        u3 = Conv2d(u3, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3_aux/c1')
        u3 = BatchNormLayer(u3, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3_aux/b1')
        u3 = PadLayer(u3, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3_aux/pad2')
        u3 = Conv2d(u3, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u3_aux/c2')
        u3 = BatchNormLayer(u3, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u3_aux/b2')
        u3 = u3.outputs

        n = UpSampling2dLayer_(n, (2, 2), method = 1, align_corners=True, name='u2/u')
        n = ConcatLayer([n, d2], concat_dim = 3, name='u2/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad2')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b3')

        u2 = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2_aux/pad1')
        u2 = Conv2d(u2, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2_aux/c1')
        u2 = BatchNormLayer(u2, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2_aux/b1')
        u2 = PadLayer(u2, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2_aux/pad2')
        u2 = Conv2d(u2, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u2_aux/c2')
        u2 = BatchNormLayer(u2, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u2_aux/b2')
        u2 = u2.outputs

        n = UpSampling2dLayer_(n, (2, 2), method = 1, align_corners=True, name='u1/u')
        n = ConcatLayer([n, d1], concat_dim = 3, name='u1/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad3')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b3')

        u1 = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1_aux/pad1')
        u1 = Conv2d(u1, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1_aux/c1')
        u1 = BatchNormLayer(u1, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1_aux/b1')
        u1 = PadLayer(u1, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1_aux/pad2')
        u1 = Conv2d(u1, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u1_aux/c2')
        u1 = BatchNormLayer(u1, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u1_aux/b2')
        u1 = u1.outputs

        n = UpSampling2dLayer_(n, (2, 2), method = 1, align_corners=True, name='u0/u')
        n = ConcatLayer([n, d0], concat_dim = 3, name='u0/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad_init')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c_init')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b_init')
        gan_feat = n.outputs

        u0 = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0_aux/pad1')
        u0 = Conv2d(u0, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0_aux/c1')
        u0 = BatchNormLayer(u0, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0_aux/b1')
        u0 = PadLayer(u0, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0_aux/pad2')
        u0 = Conv2d(u0, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u0_aux/c2')
        u0 = BatchNormLayer(u0, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u0_aux/b2')
        u0 = u0.outputs

        refine_lists = []
        refine_lists.append(n.outputs)
        for i in np.arange(7):
            n_res = n
            n_res = Conv2d(n_res, 64, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c_res{}'.format(i))#
            n_res = BatchNormLayer(n_res, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b_res{}'.format(i))#

            n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad{}_1'.format(i))
            n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c{}_1'.format(i))
            n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b{}_1'.format(i))
            n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad{}_2'.format(i))
            n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c{}_2'.format(i))
            n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b{}_2'.format(i))
            n = ElementwiseLayer([n, n_res], tf.add, name='u0/add{}'.format(i))#
            refine_lists.append(n.outputs)

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad1')#
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c1')#
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b1')#
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad2')#
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c2')#
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b2')#
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad3')#pad1
        n = Conv2d(n, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='uf/c3')#c1

        return tf.nn.sigmoid(n.outputs), [u4, u3, u2, u1, u0], gan_feat, refine_lists

def discriminator(feats, is_train=True, reuse=False, scope = 'discriminator'):
    w_init = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    b_init = None
    g_init = None
    
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse):
        n = InputLayer(feats, name='input_feature')

        n = Conv2d(n, 64, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h0/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h0/b1')#
        n = Conv2d(n, 128, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h1/b1')#
        n = Conv2d(n, 256, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h2/b1')#
        n = Conv2d(n, 512, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h3/b1')#
        n = Conv2d(n, 1, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init_sigmoid, b_init=b_init, name='h4/c1')

        logits = n.outputs

    return logits, tf.nn.sigmoid(logits)



def deblur_net(input_defocus_map,input_defocus_img,is_train=False, reuse=False, scope = 'deblur_net'):

    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    b_init = None
    g_init = None
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    def UpSampling2dLayer_(input, scale, method, align_corners, name):
        input = input.outputs
        tensor_size = input.get_shape().as_list()# tf.shape(input)
        n = InputLayer(input, name = name + '_in')
        new_size = [int(tensor_size[1] * scale[0]), int(tensor_size[2] * scale[1])]
        n = UpSampling2dLayer(n, size=new_size, is_scale = False, method = method, align_corners = align_corners, name = name)

        return n

    def MultipleLayers(dfm_layer, df_img_layer, max_coc, name):
        dfm_rngs = tf.linspace(0.0, 1.0, int(max_coc), name=name+'_generate_brightness_ranges')
        for idx in range(max_coc):
            sel_rng = dfm_rngs[idx]
            dfm_dist = tf.subtract(tf.constant(1.0), tf.abs(tf.subtract(dfm_layer.outputs, sel_rng)))
            masked_dfm_img = tf.multiply(df_img_layer.outputs, dfm_dist)

            if idx == 0:
                concat_input = masked_dfm_img
            else:
                concat_input = tf.concat([concat_input, masked_dfm_img], axis = -1)

        concat_input_layer = InputLayer(concat_input, name=name+'_concatneted_input_layer')
        
        return concat_input_layer

    def MDP(input, name):
        n0_1 = Conv2d(input, 64, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n0_1')
        n0_2 = Conv2d(n0_1, 128, (3,3), (2,2), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n0_2')

        n1_1 = Conv2d(n0_2, 128, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n1_1')
        n1_2 = Conv2d(n1_1, 128, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n1_2')
        n1_3 = Conv2d(n1_2, 256, (3,3), (2,2), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n1_3')

        n2_1 = Conv2d(n1_3, 256, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n2_1')
        n2_2 = Conv2d(n2_1, 256, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n2_2')
        n2_3 = Conv2d(n2_2, 512, (3,3), (2,2), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n2_3')

        n3_1 = Conv2d(n2_3, 512, (3,3), (1,1), padding='same', dilation_rate=(2,2), W_init=w_init_relu, act=lrelu, name=name+'n3_1')
        n3_2 = Conv2d(n3_1, 512, (3,3), (1,1), padding='same', dilation_rate=(2,2), W_init=w_init_relu, act=lrelu, name=name+'n3_2')
        n3_3 = Conv2d(n3_2, 512, (3,3), (1,1), padding='same', dilation_rate=(2,2), W_init=w_init_relu, act=lrelu, name=name+'n3_3')

        n4 = ConcatLayer([n3_3, n2_3], name = name+'n4_cat_n3_3_n2_3')
        n4_1 = Conv2d(n4, 256, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n4_1')
        n4_1 = UpSampling2dLayer_(n4_1, (2,2), method=0, align_corners= False, name=name+'n4_upsampling')
        n4_2 = Conv2d(n4_1, 256, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n4_2')
        n4_3 = Conv2d(n4_2, 256, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n4_3')

        n5 = ConcatLayer([n4_3, n1_3], name = name+'n5_cat_n4_3_n1_3')
        n5_1 = Conv2d(n5, 128, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n5_1')
        n5_1 = UpSampling2dLayer_(n5_1, (2,2), method=0, align_corners= False, name=name+'n5_upsampling')
        n5_2 = Conv2d(n5_1, 128, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n5_2')

        n6 = ConcatLayer([n5_2, n0_2], name = name+'n6_cat_n5_2_n0_2')
        n6_1 = Conv2d(n6, 64, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n6_1')
        n6_1 = UpSampling2dLayer_(n6_1, (2,2), method=0, align_corners= False, name=name+'n6_upsampling')
        n6_2 = Conv2d(n6_1, 64, (3,3), (1,1), padding='same', W_init=w_init_relu, act=lrelu, name=name+'n6_2')

        n_output = Conv2d(n6_2, 3, (3,3), (1,1), act=tf.nn.sigmoid, padding='same', W_init=w_init_sigmoid, name=name+'n_out')

        return n_output

    with tf.variable_scope(scope, reuse=reuse):

        input_df_map_layer = InputLayer(input_defocus_map, name='input_defocus_map')
        input_df_img_layer = InputLayer(input_defocus_img, name='input_defocus_img')

        concat_input_layer = MultipleLayers(input_df_map_layer, input_df_img_layer, max_coc, name='input_layer')
        n_out = MDP(concat_input_layer, name = 'MDP_ds_0_')
        
    return n_out

