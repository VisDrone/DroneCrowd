% drawss bbox on an image (im1) and returns the image file
% bbox: a matrix of size n*5
% default line width (lw) is 2
function im1 = show_bbox_on_image(im1, bbox, bws, col, lw)

if ~exist('lw')
  lw = 2;
end

m1 = floor((lw-1)/2);   %% reduce 1 for the pixel itself
m2 = ceil((lw-1)/2);

[sz1 sz2 sz3] = size(im1);
sz = size(bbox, 1);

bbox = round(bbox);

for j = floor(size(bbox,2)/4):-1:1 %%for all parts
  for i = 1:sz
    x1 = bbox(i, (j-1)*4+1);
    y1 = bbox(i, (j-1)*4+2);
    x2 = bbox(i, (j-1)*4+3);
    y2 = bbox(i, (j-1)*4+4);
    
    for k = 1:3  %% RGB channels
      im1(max(1,y1-m1):min(sz1,y1+m2),  max(1,x1):min(sz2,x2),        k) = col(k, bbox(i,end));
      im1(max(1,y2-m1):min(sz1,y2+m2),  max(1,x1):min(sz2,x2),        k) = col(k, bbox(i,end));
      im1(max(1,y1):min(sz1,y2),        max(1,x1-m1):min(sz2,x1+m2),  k) = col(k, bbox(i,end));
      im1(max(1,y1):min(sz1,y2),        max(1,x2-m1):min(sz2,x2+m2),  k) = col(k, bbox(i,end));
    end
    if ~isempty(bws)  %% add text if needed
      col1  = col(:, bbox(i, end));
      im1 = show_text_on_image(im1, num2str(bbox(i,end)), col1, min(max(x1-10,1),sz2), min(max(y1-20,1),sz1), 20, bws(bbox(i,end)).bw);
    end
  end
end

function im = show_text_on_image(im, txt, col, x, y, h, bw)
[sz11 sz22 sz33] =size(im);

[sz1 sz2] = size(bw);
y2 = min(y+sz1-1, sz11);
x2 = min(x+sz2-1, sz22);

for k = 1:3 %% RGB channels
  im(y2-sz1+1:y2, x2-sz2+1:x2, k) = (1-bw) * col(k);
end


