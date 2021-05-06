function [bleft bright btop bbottom]= clampBBox(bleft, bright, btop, bbottom, imgWidth, imHeight)
% 
% (C) Anton Andriyenko, 2012
%
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without written permission from the authors.

    
% clamp bounding box to [1, imageDim]

bleft=max(1,bleft); bleft=min(imgWidth,bleft);
bright=max(1,bright); bright=min(imgWidth,bright);
btop=max(1,btop); btop=min(imHeight,btop);
bbottom=max(1,bbottom); bbottom=min(imHeight,bbottom);

end