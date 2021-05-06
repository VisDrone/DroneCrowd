function [missr, fppi] = score(c, g, n)

thr = 0.5;

if isempty(c) || isempty(g),
  missr = 1;
  fppi  = inf;
  prec  = 0;
  rec   = 0;
  ap    = 0;
  return;
end

r   = c.r;
pos = zeros(length(c.x), 1);

%%% find positives
for fr = min(c.fr):max(c.fr)
  I = find(c.fr == fr);
  J = find(g.fr == fr);
  [v,ind] = sort(r(I));
  I = I(ind);
  
  while ~isempty(I) && ~isempty(J),
    i = I(end);                   %% Select highest scoring candidate
    ov = calc_overlap(c,i,g,J);   %% Search for an unclaimed positive
    [val, ind] = max(ov);
    if val > thr,
      pos(i) = 1;
      J(ind) = [];
    end
    I(end) = [];
  end
end

% remove false-positives that overlap with "people" in ground truth
keep = logical(ones(size(c.x)));
for fr = min(c.fr):max(c.fr)
  I = find(c.fr == fr & pos == 0);
  J = find(n.fr == fr);
  [v,ind] = sort(r(I));
  I = I(ind);
  
  while ~isempty(I) && ~isempty(J),
    i = I(end);                   %% Select highest scoring candidate
    ov = calc_overlap(c,i,n,J);   %% Search for a positive on "people" area.
    [val,ind] = max(ov);
    if val > thr,
      keep(i) = 0;    %% ignore it completely since it overlaps with "people" 
      J(ind) = [];
    end
    I(end) = [];
  end
end

r   = r(keep);
pos	= pos(keep);

[r, I]  = sort(r,'descend');
pos     = pos(I);
fp      = cumsum(~pos);
tp      = cumsum(pos);
rec     = tp/length(g.x);
prec    = tp./(fp+tp);

nfrs  = length(unique(c.fr));
% nfrs = max(c.fr) - min(c.fr) + 1;
missr = 1-rec;
fppi  = fp / nfrs;

