# Author: Deepak Pathak (c) 2016
# Modifications: 
# 29/03/19:
#  * added conditions to work with grayscale images w/o changing the code each time.
#  * pass any two images as input (first 2 arguments)
#  * pass output folder (name is the same as img1)
#  * fixed visualization code for grayscale images (make it independent of the number of channels in the input as we always want to output a coloured png!)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import os
import pyflow

parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
# Mandatory parameters (path to img1-img2)
parser.add_argument("im1_path", default='examples/car1.jpg', help='Path to first image')
parser.add_argument("im2_path", default='examples/car2.jpg', help='Path to second image')
parser.add_argument("out_dir", default='examples', help='Path where the results will be stored')
parser.add_argument("colour", default='1', help='whether we are working with rgb or gray images')


# Optional parameters
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()

if int(args.colour) == 0:  # read as grayscale
  colType = 1
  # Actually Luminance + Alpha channel (transparency)
  im1 = np.array(Image.open(args.im1_path).convert('LA'))
  im2 = np.array(Image.open(args.im2_path).convert('LA'))
else:
  colType = 0
  im1 = np.array(Image.open(args.im1_path))
  im2 = np.array(Image.open(args.im2_path))

im1 = im1.astype(float) / 255.
im2 = im2.astype(float) / 255.

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30

s = time.time()
u, v, im2W = pyflow.coarse2fine_flow(
    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)
e = time.time()
print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
flow = np.concatenate((u[..., None], v[..., None]), axis=2)
img1_name = args.im1_path.split('/')[-1]
out_np_name = "{0}_out{1}".format(img1_name[:-4], '.npy') 
np.save(os.path.join(args.out_dir, out_np_name), flow)

if args.viz:
    import cv2
    hsv = np.zeros((im1.shape[0], im1.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    out_rgb_name = "{0}_out{1}".format(img1_name[:-4], '.png')
    cv2.imwrite(os.path.join(args.out_dir, out_rgb_name), rgb)
    out_rgb_w_name = out_rgb_name.replace('.png', '_warped.jpg')
    # Using ::-1 to convert manually from RGB to BGR
    # This only works with colour images and is more risky than using opencv's cvtColor(...)
    if int(args.colour) == 1:
      # cv2.imwrite(os.path.join(args.out_dir, out_rgb_w_name), im2W[:, :, ::-1] * 255)
      cv2.imwrite(os.path.join(args.out_dir, out_rgb_w_name), cv2.cvtColor(im2W.astype('uint8'), cv2.COLOR_RGB2BGR))
    elif int(args.colour) == 0:
      cv2.imwrite(os.path.join(args.out_dir, out_rgb_w_name), cv2.cvtColor(im2W[:,:,0].astype('uint8'), cv2.COLOR_GRAY2BGR))
    else:
      print("FATAL: wrong number of channels (expected grayscale/rgb)")

