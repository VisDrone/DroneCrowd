function s = add(s1,s2),
% s = add(s1,s2)
% Appends structures s2 to the end of s1
  
if isempty(s1)
  s = s2;
elseif isempty(s2),
  s = s1;
else
  n = fieldnames(s1);
  for i = 1:length(n),
    f = n{i};
    s.(f) = cat(1,s1.(f),s2.(f));
  end
end
