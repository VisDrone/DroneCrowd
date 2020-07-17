import scipy
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
import json
import torch
from multiprocessing import Pool
from functools import partial


def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))

    print 'generate density...'
    for i, pt in enumerate(pts):
        sigma = 5
        draw_msra_gaussian(density, pt, sigma)
    print 'done.'
    return density

def process(idx, img_paths):
	img_path = img_paths[idx]
	mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('img','GT_img'))
	img = plt.imread(img_path)

	k = np.zeros((int(img.shape[0]/2),int(img.shape[1]/2)))
        ids = np.zeros((int(img.shape[0]/2),int(img.shape[1]/2)), np.int16)
	gt = mat["image_info"][0,0][0,0][0]
	for i in range(0,len(gt)):
	    if int(gt[i][1]/2)<int(img.shape[0]/2) and int(gt[i][0]/2)<int(img.shape[1]/2):
	        k[int(gt[i][1]/2),int(gt[i][0]/2)]=1
	        ids[int(gt[i][1]/2),int(gt[i][0]/2)]=gt[i][2]
	k = gaussian_filter_density(k)

	with h5py.File(img_path.replace('.jpg','_max.h5').replace('images','ground_truth'), 'w') as hf:
		    hf['location'] = k
                    hf['identity'] = ids

part_train = os.path.join('train_data','images')
part_test = os.path.join('val_data','images')

img_paths = []
for img_path in glob.glob(os.path.join(part_test, '*.jpg')):
	img_paths.append(img_path)
for img_path in glob.glob(os.path.join(part_train, '*.jpg')):
	img_paths.append(img_path)

pool = Pool(16)
partial = partial(process, img_paths=img_paths)
_ = pool.map(partial, range(len(img_paths)))
