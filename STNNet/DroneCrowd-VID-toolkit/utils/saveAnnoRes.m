function [allgt, alldet] = saveAnnoRes(gtPath, detPath, listPath)
%% process the annotations and groundtruth
seqList = load(listPath);
numSeqs = length(seqList);
allgt = cell(1,numSeqs);
alldet = cell(1,numSeqs);

for idSeq = 1:numSeqs
    seqID = seqList(idSeq);
    seqName = sprintf('%05d',seqID);
    tmp = load(fullfile(gtPath, [seqName '.mat']));
    anno = tmp.anno;
    gt_ = [anno(:,1)+1, anno(:,2:6), zeros(size(anno,1), 1)]; 
    det = [];
    gt = [];
    for k = 1:300
        curdet = load(fullfile(detPath, sprintf('img%03d%03d_loc.txt',seqID,k)));
        idx = gt_(:,1) == k;
        gt = cat(1, gt, gt_(idx,:));
        if(size(curdet,2) ~= 3)
            curdet = curdet';
        end
        numdet = size(curdet,1);
        if(numdet>0)
            curdet = [repmat([k, -1], [numdet, 1]), curdet(:,1)-10, curdet(:,2)-10, repmat([20, 20], [numdet, 1]), curdet(:,3)];
            det = cat(1, det, curdet);
        end 
    end
    allgt{idSeq} = gt;
    alldet{idSeq} = det;
end