function dres = bboxes2dres(bboxes)
k=0;

dres.x = [];
dres.y = [];
dres.w = [];
dres.h = [];
dres.r = [];
dres.fr = [];

for i=1:length(bboxes)
  bbox = bboxes(i).bbox;
  if isempty(bbox)
    continue
  end
  dres.x = [dres.x; bbox(:,1)];
  dres.y = [dres.y; bbox(:,2)];
%   dres.w = [dres.w; bbox(:,3)];
%   dres.h = [dres.h; bbox(:,4)];
  dres.w = [dres.w; bbox(:,3) - bbox(:,1)+1];
  dres.h = [dres.h; bbox(:,4) - bbox(:,2)+1];
  dres.r = [dres.r; bbox(:,5)];
  dres.fr = [dres.fr; repmat(i, [size(bbox,1) 1])];
end

    
    
    

