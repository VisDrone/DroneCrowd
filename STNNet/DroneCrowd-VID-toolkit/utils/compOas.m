function dis = compOas(dt, gt)
m = size(dt,1); 
n = size(gt,1); 
dis = zeros(m,n);

for i = 1:m
    posdet = [dt(i,1) + dt(i,3)/2, dt(i,2) + dt(i,4)/2];
    for j = 1:n
        posgt = [gt(j,1) + gt(j,3)/2, gt(j,2) + gt(j,4)/2];
        dis(i,j) = sqrt((posdet(1)-posgt(1))^2+(posdet(2)-posgt(2))^2);
    end
end