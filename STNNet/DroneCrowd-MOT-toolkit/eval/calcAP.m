function [aps, recall, precision] = calcAP(track_confs, tp_cell, fp_cell, num_vids, num_track_per_class, num_track_thr, defaultTrackThr)

fprintf('computing AP\n');
recall = cell(1,num_track_thr);
precision = cell(1,num_track_thr);
aps = cell(1,num_track_thr);
confs = [track_confs{:}];
[~, ind] = sort(confs,'descend');
for o = 1:num_track_thr
    tp_all = [];
    fp_all = [];
    for v = 1:num_vids
        tp_all = [tp_all(:); tp_cell{v}{o}'];
        fp_all = [fp_all(:); fp_cell{v}{o}'];
    end
    
    tp_all = tp_all(ind)';
    fp_all = fp_all(ind)';
    
    % compute precision/recall
    tp = cumsum(tp_all);
    fp = cumsum(fp_all);
    recall{o} = (tp/num_track_per_class)';
    precision{o} = (tp./(fp+tp))';
    aps{o} = VOCap(recall{o},precision{o})*100;
end

fprintf('-------------\n');
ap = aps{1};
for t = 2:length(aps)
    ap = ap + aps{t};
end
ap = ap ./ length(aps);
fprintf('Mean AP:\t\t %0.2f%%\n',mean(ap));
fprintf(' = = = = = = = = \n');
for t = 1:length(aps)
    ap = aps{t};
    fprintf('Mean AP@%0.2f:\t %0.2f%%\n',defaultTrackThr(t),mean(ap));
end
fprintf(' = = = = = = = = \n');