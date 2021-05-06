function gt = greedy_gt_generator(gtInfo)

count = 1;
[r, c] = size(gtInfo.X);
for i = 1:r
    for j = 1:c
        if(gtInfo.W(i,j) ~= 0 && gtInfo.H(i,j) ~= 0)
            gt.x(count,1) = gtInfo.X(i,j)-0.5*gtInfo.W(i,j);
            gt.y(count,1) = gtInfo.Y(i,j)-gtInfo.H(i,j);
            gt.w(count,1) = gtInfo.W(i,j);
            gt.h(count,1) = gtInfo.H(i,j);
            gt.fr(count,1) = i;
            gt.vid(count,1) = 3;
            count = count+1;
        end
    end
end