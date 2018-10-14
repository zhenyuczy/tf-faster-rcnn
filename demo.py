#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'car')


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.2f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_path, vis):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(image_path)

    image_name = image_path[image_path.rfind('/') + 1:]

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    '''
    score: A numpy array with shape (N, C), where N is the number of object proposals and 
            C is the number of classes
    boxes: A numpy array with shape (N, C*4)
    '''
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.80
    NMS_THRESH = 0.3

    temp_answer = []
    temp_answer.append(image_name)
    '''
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)    
    return None
    '''
    cls_ind, cls = 1, 'car'
    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    '''
    dets: A numpy array with shape (N, C*4+1), the last column is score.
    '''
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    if vis == 'Yes':
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    boxes = np.copy(dets[inds][:, :4])
    boxes = np.round(boxes).astype(int)
    boxes_answer = []
    for box in boxes.tolist():
        box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
        boxes_answer.append('_'.join(str(coor) for coor in box))
    temp_answer.append(';'.join(box for box in boxes_answer))
    return temp_answer


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101, res152]',
                        default='res101')
    parser.add_argument('--ite', dest='demo_ite', help='The iterations of the network',
                        default='70000')
    parser.add_argument('--dir', dest='demo_dir', help='Dataset Directory for testing',
                        default='data/demo')
    parser.add_argument('--vis', dest='demo_vis', help='Whether to visualize the images',
                        default='No')
    parser.add_argument('--wri', dest='write_ans', help='Whether to write answer into csv file',
                        default='No')
    parser.add_argument('--set', dest='dataset', help='Trained dataset',
                        default='pascal_voc')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    demoite = args.demo_ite
    demodir = args.demo_dir
    demovis = args.demo_vis
    dataset = args.dataset
    write = args.write_ans

    if dataset == 'pascal_voc':
        tfmodel = os.path.join('output', demonet, 'voc_2007_trainval', 'default',
                               demonet + '_faster_rcnn_iter_' + str(demoite) + '.ckpt')
    else:  # pascal_voc_0712
        tfmodel = os.path.join('output', demonet, 'voc_2007_trainval+voc_2012_trainval',
                               'default',
                               demonet + '_faster_rcnn_iter_' + str(demoite) + '.ckpt')

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    elif demonet == 'res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 2,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    im_names = os.listdir(demodir)
    answer = []
    answer.append(['name', 'coordinate'])
    for index, im_name in enumerate(im_names):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('{}th image {}/{}'.format(index+1, demodir, im_name))
        temp_answer = demo(sess, net, os.path.join(demodir, im_name), demovis)
        answer.append(temp_answer)
    if write == 'Yes':
        import pandas as pd
        ans_pd = pd.DataFrame(answer)
        ans_pd.to_csv('answer.csv', header=None, index=None)
    print('OK!')
    if demovis == 'Yes':
        plt.show()
