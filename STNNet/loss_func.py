import torch
import math
import numpy as np
import torch.nn as nn
from pointconv import square_distance

def calc_loc_loss(criterion, locmap, regmap, reggt, loc_ratio, scale_ratio, gt):
    conf_loss = torch.Tensor(1,1).zero_().cuda() # classification loss for localization branch
    reg_loss = torch.Tensor(1,1).zero_().cuda() # regression loss for localization branch
    all_loss = torch.Tensor(1,1).zero_().cuda() # overall loss
    locmap = nn.Softmax(dim=1)(locmap)
    locmap = torch.clamp(locmap, max = 1-1e-4, min = 1e-4)
    
    for k in range(locmap.shape[0]):
        pos_inds = gt[k] > 0
        pos_ids = gt[k][pos_inds]
        pos_loc = np.where(pos_inds)
        num = len(pos_ids)
        tmp = reggt[k]
        for i in range(num):
            y, x = min(locmap.shape[2]-1, int(pos_loc[0][i]/scale_ratio)), min(locmap.shape[3]-1, int(pos_loc[1][i]/scale_ratio))
            tmp[0][y,x] = 1
        posreg = torch.sum(abs(tmp), dim = 3) > 0
        posreg = posreg.type(torch.FloatTensor).cuda() 
        posmask = posreg.squeeze()   
        # calculate classification loss
        pos_num = int(torch.sum(posmask).item())
        #print(pos_num)
        if pos_num == 0:
            return all_loss, conf_loss, reg_loss

        total_num = locmap.shape[2]*locmap.shape[3]
        negmask = 1 - posmask

        pos_loss = -torch.log(locmap[k][0])*posmask
        pos_loss = torch.sum(pos_loss)
        neg_loss = -torch.log((1-locmap[k][0]))*negmask
        neg_loss, neg_idx = torch.topk(neg_loss.view(1,total_num), pos_num*3)
        neg_loss = torch.sum(neg_loss)
        conf_loss += (pos_loss+neg_loss)*loc_ratio/float(4*pos_num)/4.

        # calculate regression loss
        reg_y, gt_y = regmap[k][0,:,:]*posreg, reggt[k][:,:,:,0]*posreg
        reg_x, gt_x = regmap[k][1,:,:]*posreg, reggt[k][:,:,:,1]*posreg
        reg_y_loss = criterion(reg_y,gt_y/float(int(5./scale_ratio))).cuda() # normalization
        reg_x_loss = criterion(reg_x,gt_x/float(int(5./scale_ratio))).cuda() # normalization
        reg_loss += (reg_y_loss+reg_x_loss)*loc_ratio/float(pos_num*2)/4.
                
    all_loss = conf_loss + reg_loss
 
    return all_loss, conf_loss, reg_loss

def calc_trk_loss(offmap, gt1, gt2, offs, pts, use_tem):
    trk_range = 5
    neighbor_thre = (trk_range*5)**2
    gt_idmap = torch.zeros(offmap.shape[0], 2, offmap.shape[2], offmap.shape[3])
    pts = pts.type(torch.float)
    neighbors = square_distance(pts, pts)
    pts = pts.type(torch.long)
    all_pt_dist, all_gp_dist = 0., 0.
    pt_num, hit_num, match_num, gp_num = 0., 0., 0., 0.

    for k in range(offmap.shape[0]):
        gt_dict = {}
        pos_inds1 = gt1[k] > 0
        pos_inds2 = gt2[k] > 0
        pos_ids1 = gt1[k][pos_inds1]
        pos_ids2 = gt2[k][pos_inds2]
        pos_loc1 = np.where(pos_inds1)
        pos_loc2 = np.where(pos_inds2)
        num1 = len(pos_ids1)
        num2 = len(pos_ids2)
        h, w = offmap.shape[2] - 1, offmap.shape[3] - 1
        pt_num += num1+num2
        if num1 > 0 and num2 > 0:
            for i in range(num1):
                for j in range(num2):
                    # if ids1 appears in frame 2
                    if pos_ids1[i] == pos_ids2[j]:
                        y1, x1 = int(pos_loc1[0][i]), int(pos_loc1[1][i])
                        y2, x2 = int(pos_loc2[0][j]), int(pos_loc2[1][j])
                        left, right = max(0, x1 - trk_range), min(w, x1 + trk_range)
                        top, down = max(0, y1 - trk_range), min(h, y1 + trk_range)
                        gt_idmap[k, 0, top:down+1, left:right+1] = pos_ids1[i]
                        left, right = max(0, x2 - trk_range), min(w, x2 + trk_range)
                        top, down = max(0, y2 - trk_range), min(h, y2 + trk_range)
                        gt_idmap[k, 1, top:down+1, left:right+1] = pos_ids2[j]
                        gt_dict[str(int(pos_ids1[i]))] = [x1, y1, x2, y2]
                        match_num += 1.
                # if ids1 disappears in frame 2
                if str(int(pos_ids1[i])) not in gt_dict.keys():
                    gt_dict[str(int(pos_ids1[i]))] = []

        # check individual point
        matched_ids = []
        pt_dict = {}
        for i in range(pts.shape[1]):
            cur_i, cur_j = pts[k, i, :]
            w, h = gt_idmap.shape[3] - 1, gt_idmap.shape[2] - 1
            cur_id = gt_idmap[k, 0, int(max(min(h,cur_i),0)), int(max(min(w,cur_j),0))].unsqueeze(0) # check current id
            if cur_id > 0: # ignore the unmatched points
                if str(int(cur_id)) not in gt_dict.keys():# ignore the matched points only at current id
                    hit_num += 1.
                    continue
                if not (cur_id in matched_ids): # if matched
                    hit_num += 1.
                    gt_correspond = gt_dict[str(int(cur_id))]
                    matched_ids.append(cur_id)
                    # add time offset at current frame
                    nex_i, nex_j = cur_i - offs[k, i, 0]*5., cur_j - offs[k, i, 1]*5.
                    nex_id = gt_idmap[k, 1, int(max(min(h,nex_i),0)), int(max(min(w,nex_j),0))].unsqueeze(0) # check next id
                    if nex_id > 0: # if matched at both current id and next id
                        hit_num += 1.
                        if nex_id == cur_id:
                            all_pt_dist += (abs(nex_j - gt_correspond[2])/5. + abs(nex_i - gt_correspond[3])/5.)/2.
                            pt_dict[str(i)] = [nex_j, nex_i, gt_correspond[2], gt_correspond[3]]
                    else: # if matched at only current id
                        all_pt_dist += 2.
                        pt_dict[str(i)] = [0,0,0,0]
                else: # if matched at current id twice (false positives)
                    all_pt_dist += 2.
                    pt_dict[str(i)] = [0,0,0,0]
        if not use_tem:
        # check neighboring points
            frame_dist = 0.
            for i in range(pts.shape[1]):
                if str(i) in pt_dict.keys(): # ignore unmatched pairs
                    dist = []
                    gp_num += 1.
                    target_correspond = pt_dict[str(i)]
                    if target_correspond[0] != 0: # if point i corresponded
                        cur_neighbors = neighbors[k,i,:]
                        neighbor_idx = np.where(cur_neighbors.cpu().numpy() <= neighbor_thre)
                        num_neighbors = len(neighbor_idx[0])
                        for j in range(num_neighbors):
                            if cur_neighbors[neighbor_idx[0][j]] > 0 and str(neighbor_idx[0][j]) in pt_dict.keys(): # not the same point
                                supp_correspond = pt_dict[str(neighbor_idx[0][j])]
                                if supp_correspond[0] != 0: # if point j corresponded
                                    vec1 = [supp_correspond[0]-target_correspond[0],supp_correspond[1]-target_correspond[1]]
                                    vec2 = [supp_correspond[2]-target_correspond[2],supp_correspond[3]-target_correspond[3]]
                                    cur_dist = (abs(vec1[0]-vec2[0])/5. + abs(vec1[1]-vec2[1])/5.)/2.  
                                    dist.append(cur_dist)
                    if len(dist):
                        frame_dist += sum(dist)/len(dist)
            all_gp_dist += frame_dist

    all_recall = hit_num/max(1.0, pt_num)
    all_precision = hit_num/float(2*pts.shape[1]*offmap.shape[0])
    all_pt_dist = all_pt_dist/max(1.0,match_num)
    all_gp_dist = all_gp_dist/max(1.0,match_num)
    all_dist = (all_pt_dist + all_gp_dist)

    return all_dist, all_recall, all_precision

