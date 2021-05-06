% this function write contents of "txt" on an image and then returns its image.
% "h" is the texyt height in pixels
% "figNum" is the number for figure.
% note that you may not interacti with matlab while running this command since the figure window should be on the top.

function bw = text_to_image(txt, h, figNum)
f = h/1.2;
sz1 = round(f*4);
sz2 = round(length(txt)*f*1.5);
r = 1;
if sz2 > 800
  r = sz2/800;
  sz2 = round(sz2/r);
  f = round(f/r);
end

im1 = ones(sz1, sz2, 3);
flag1 = 0;
if ~exist('figNum')
  figNum = ceil((rand*1000)+1);
  flag1 = 1;
end

figure(figNum);
imshow(im1);
text(10,f*3, txt, 'fontsize', f,'color', 'b');
im2 = getframe;
if flag1
  close(figNum);
end

im2 = im2.cdata;
bw2 = im2bw(im2);
f1 = find(sum(~bw2));
f2 = find(sum(~bw2'));
x1 = f1(1);
y1 = f2(1);
x2 = f1(end);
y2 = f2(end);
bw = ~bw2(y1:y2, x1:x2);

if r~=1
  bw = im2bw(imresize(double(bw), h/size(bw,1)));
end

