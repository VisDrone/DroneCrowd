function displaySeq(seqPath, listPath, allgt, alldet, isSeqDisplay) 
%% show the groundtruth and detection results
if(isSeqDisplay)
    seqList = load(listPath);
    numSeqs = length(seqList);
    ratio = 1;
    for idSeq = 1:numSeqs
        seqID = seqList(idSeq);
        gt = allgt{idSeq};
        det = alldet{idSeq};
        for idFr = 1:300
            img = imread(fullfile(seqPath, sprintf('img%03d%03d.jpg', seqID, idFr)));
            img = imresize(img, ratio);
            curgt = max(0,gt(gt(:,1) == idFr, :));
            curdet = max(0,det(det(:,1) == idFr, :));
            img = max(0, img - 20);
            figure(1),imshow(img); hold on;
            % show the points
            plot((curgt(:,3)+curgt(:,5))/2*ratio, (curgt(:,4)+curgt(:,6))/2*ratio, 'g.', 'MarkerSize', 15);            
            plot((curdet(:,3)+curdet(:,5)/2)*ratio, (curdet(:,4)+curdet(:,6)/2)*ratio, 'ro', 'MarkerSize', 8, 'LineWidth',1.5);   
            text(10,20,['#' num2str(idFr) '/300'], 'Color','red','FontSize',20);
            pause(0.01);
        end
    end
end