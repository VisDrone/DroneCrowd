function bboxes = dres2bboxes(dres, fnum)
for i = 1:fnum
  bboxes(i).bbox = [];
end

for i = 1:length(dres.x)
  bbox = [dres.x(i) dres.y(i) dres.x(i)+dres.w(i) dres.y(i)+dres.h(i) dres.id(i)];
  bboxes(dres.fr(i)).bbox = [bboxes(dres.fr(i)).bbox; bbox];
end