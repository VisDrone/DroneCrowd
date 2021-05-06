function gtdata = breakGts(gtdata)
% normalize IDs
[~, ~, ic] = unique(gtdata(:,2)); 
gtdata(:,2) = ic;
% break the groundtruth trajetory with multiple object categories
tracks = unique(ic);
newtrack_id = max(tracks) + 1;
for i = 1:numel(tracks)
    idx = gtdata(:, 2) == tracks(i);
    cls = gtdata(idx, 8);
    allcls = unique(cls);
    if(numel(allcls) > 1)
        for j = 2:numel(allcls)
            idxCls = idx & gtdata(:, 8) == allcls(j);
            gtdata(idxCls, 2) = newtrack_id;
            newtrack_id = newtrack_id + 1;
        end
    end
end  