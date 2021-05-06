function [Xgp Ygp]=projectToGroundPlane(Xi, Yi, sceneInfo)
% 
% (C) Anton Andriyenko, 2012
%
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without written permission from the authors.

[F, N]=size(Xi);
Xgp=zeros(size(Xi));
Ygp=zeros(size(Xi));


for t=1:F
    extar=find(Xi(t,:));
    for id=extar
        [Xgp(t,id), Ygp(t,id), zw]=imageToWorld(Xi(t,id), Yi(t,id), sceneInfo.camPar);
    end
end

    
end
