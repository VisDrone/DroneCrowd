function mAP = calcAccuracy(allgt, alldet)
mAP = zeros(1, 25);
    
for thr = 1:25
    if(mod(thr,5)==1)
        disp(['evaluating distance threshold ' num2str(thr) '/25...']);
    end
    gtMatch = [];
    detMatch = [];    
    for idSeq = 1:length(allgt)
        gt = allgt{idSeq};
        det = alldet{idSeq};  
        frs = unique(gt(:,1));
        if(size(det,1)>0)
            for i = 1:numel(frs)
                idxGt = gt(:, 1) == frs(i);
                idxDet = det(:, 1) == frs(i);
                gt0 = gt(idxGt,3:7);
                dt0 = max(0, det(idxDet,3:7)); 
                [gt1, dt1] = evalRes(gt0, dt0, thr);
                gtMatch = cat(1, gtMatch, gt1(:,5));
                detMatch = cat(1, detMatch, dt1(:,5:6));
            end
        end
    end 
    [~,idrank] = sort(-detMatch(:,1));
    tp = cumsum(detMatch(idrank,2)==1);
    rec = tp/max(1,numel(gtMatch));
    fp = cumsum(detMatch(idrank,2)==0);        
    prec = tp./max(1,(fp+tp));
    mAP(thr) = VOCap(rec,prec)*100;
end
disp('Evaluation Completed. The peformance of the detector is presented as follows.')