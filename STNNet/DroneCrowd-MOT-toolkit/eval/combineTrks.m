function res = combineTrks(evalClassSet, allRes)
% combine tracks of multiple object categories  
maxID = 0;
res = [];
for idClass = 1:length(evalClassSet)
    if(~isempty(allRes{idClass}))
        resClass = allRes{idClass};
        resClass(:,2) = resClass(:,2) + maxID;
        maxID = maxID + max(resClass(:,2)) + 1;  
        res = cat(1, res, resClass);
    end
end

% check the tracking results
cls = unique(res(:,8));
allids = [];
for i = 1:length(cls)
    idx = res(:,8) == cls(i);
    ids = unique(res(idx, 2));
    if(~nnz(ismember(ids, allids)))
        allids = cat(1, allids, ids);
    else
        error('The objects of multiple object categories use the same track_id.');
    end
end
