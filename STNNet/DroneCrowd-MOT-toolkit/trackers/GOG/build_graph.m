function dres = build_graph(dres)
ov_thresh = 0.3;  %original = 0.5

for fr = 2:max(dres.fr)
  f1 = find(dres.fr == fr);     %% indices for detections on this frame
  f2 = find(dres.fr == fr-1);   %% indices for detections on the previous frame
  if(isfield(dres, 'ft')) 
      feat1 = dres.ft(f1,:);
      feat2 = dres.ft(f2,:);
  end
  for i = 1:length(f1)
    ovs1  = calc_overlap(dres, f1(i), dres, f2);
    if(isfield(dres, 'ft')) 
        dist = pdist2(feat1(i,:),feat2);
        inds1 = find(ovs1 > ov_thresh & dist < 5);                       %% find overlapping bounding boxes.  
    else
        inds1 = find(ovs1 > ov_thresh);
    end
    
    ratio1 = dres.h(f1(i))./dres.h(f2(inds1));
    inds2  = (min(ratio1, 1./ratio1) > 0.8);          %% we ignore transitions with large change in the size of bounding boxes.

    dres.nei(f1(i),1).inds  = f2(inds1(inds2))';      %% each detction window will have a list of indices pointing to its neighbors in the previous frame.
  end
end