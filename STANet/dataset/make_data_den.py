import h5py
import scipy
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from scipy.ndimage.filters import gaussian_filter 
from scipy import spatial
import json
import torch
from multiprocessing import Pool
from functools import partial


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    leafsize = 2048
    # build kdtree
    tree = spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print 'generate density...'
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print 'done.'
    return density

def process(idx, img_paths):
	img_path = img_paths[idx]
	mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('img','GT_img'))
	img = plt.imread(img_path)
	k = np.zeros((img.shape[0],img.shape[1]))
	gt = mat["image_info"][0,0][0,0][0]
	for i in range(0,len(gt)):
	    if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
	        k[int(gt[i][1]),int(gt[i][0])]=1
	k = gaussian_filter_density(k)

	with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
		    hf['density'] = k


part_train = os.path.join('train_data','images')
part_test = os.path.join('val_data','images')

img_paths = []
for img_path in glob.glob(os.path.join(part_train, '*.jpg')):
	img_paths.append(img_path)
for img_path in glob.glob(os.path.join(part_test, '*.jpg')):
	img_paths.append(img_path)

pool = Pool(16)
partial = partial(process, img_paths=img_paths)
_ = pool.map(partial, range(len(img_paths)))
