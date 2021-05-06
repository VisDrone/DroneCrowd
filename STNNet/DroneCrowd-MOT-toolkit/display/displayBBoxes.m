function displayBBoxes(sceneInfo, stateInfo)
% Draw bounding boxes on top of images
% 
% (C) Anton Andriyenko, 2012
%
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without written permission from the authors.
%

X = stateInfo.X;
Y = stateInfo.Y;
W = stateInfo.W;
H = stateInfo.H;

[F, numObjs] = size(X);
if(F == 0)
    disp('no tracking results!');
end
ind=find(W);
aspectRatio=mean(H(ind)./W(ind));
colors = getColorSet(numObjs);

for t = 1:3:F
    clf
    im = imread(sceneInfo.frames{t});
    im = (double(im)-20)/255;
    imshow(im,'Border','tight'), hold on;
    
    % frame number
    text(10,20,sprintf('#%04d',t),'FontSize',30,'color','r');
    
    extar=find(X(t,:));
    % foot position
    if(sceneInfo.displayDots)
        for id=extar
            plot(X(t,id),Y(t,id),'.','color',colors(id,:),'MarkerSize',sceneInfo.dotSize);
        end
    end
    
    % box
    if(sceneInfo.displayBoxes)
        for id=extar
            bleft=X(t,id)-W(t,id)/2;
            btop=Y(t,id)-H(t,id);
            rectangle('Position',[bleft,btop,W(t,id),H(t,id)],'Curvature',[.1,.2*(W(t,id)/H(t,id))],'EdgeColor',colors(id,:),'linewidth',sceneInfo.boxLineWidth);
        end
    end
    
    % ID
    if sceneInfo.displayID
        for id=extar
            tx=X(t,id)-W(t,id)/2; 
            ty=Y(t,id)-H(t,id)*0.8; % on top
            text(tx,ty,sprintf('%i',id),'color',colors(id,:), 'HorizontalAlignment','left', 'FontSize',W(t,id)/5, 'FontUnits','pixels','FontWeight','bold');
        end
    end
    
    % cropouts
    if(sceneInfo.displayCropouts)
        bw=2; %border cropouts
        
       %% crop outs var sized
        maxTar=30;
        extarRed=extar(extar<=maxTar); % reducde

       %% crop outs fixed sized
        uniH=min(60,round(sceneInfo.imgHeight/10)); uniW=round(uniH/aspectRatio);
        mxfac=.5;
        crpImg=im(1:uniH,1:min(sceneInfo.imgWidth,round(uniW*size(W,2)+bw*size(W,2))),:)*mxfac; 
        crpImg=crpImg + (1-mxfac)*ones(size(crpImg));   % bleeched
        for id=extarRed
            offset=(id-1)*uniW + (id-1)*bw+1;
            bleft=round(X(t,id)-W(t,id)/2);
            bright=round(X(t,id)+W(t,id)/2);
            btop=round(Y(t,id)-H(t,id));
            bbottom=round(Y(t,id));
            
            [bleft, bright, btop, bbottom] = clampBBox(bleft, bright, btop, bbottom, sceneInfo.imgWidth, sceneInfo.imgHeight);
            ht=uniH;
            imres=imresize(im(btop:bbottom,bleft:bright,:),[uniH uniW]);
            crpImg(1:ht,offset:offset+uniW-1,:)=imres;
            
        end        
        imshow(crpImg);

        for id=extarRed
            tx = id*uniW-uniW/2 + id*bw;
            ty = 30; % top
            text(tx,ty,sprintf('%i',id),'color',getColorFromID(id, numObjs), 'HorizontalAlignment','center', ...
                'FontSize',uniW/4, 'FontUnits','pixels','FontWeight','bold'); % fixed size            
        end
        if(sceneInfo.displayConnections)
            for id=extarRed
                if(t-find(X(:,id),1,'first')<5)
                btop=round(Y(t,id)-H(t,id));
                offset=(id-1)*uniW + (id-1)*bw+1 + uniW/2;
                line([X(t,id) offset],[btop uniH],'color',getColorFromID(id, numObjs),'linestyle','-');
                end
            end
        end
    end
    
    % show trace
    if(sceneInfo.traceLength)
        for tracet=max(1,t-sceneInfo.traceLength):max(1,t-1)
            ipolpar=(t-tracet)/sceneInfo.traceLength; % parameter [0,1] for color adjustment
            extarpast=find(X(tracet,:));
            % foot position
            for id=extarpast             
                if(W(tracet+1,id))
                    endcol=sceneInfo.grey;
                    line(X(tracet:tracet+1,id) ,Y(tracet:tracet+1,id), 'color',ipolpar*endcol + (1-ipolpar)*colors(id,:),'linewidth',(1-ipolpar)*sceneInfo.traceWidth+1);
                end
            end
        end
    end    
    pause(0.001);  
end