function [metrics, metricsInfo]=CLEAR_MOT3D(gtInfo,stateInfo,options)
% compute CLEAR MOT and other metrics
%
% metrics contains the following
% [1]   recall	- recall = percentage of detected targets
% [2]   precision	- precision = percentage of correctly detected targets
% [3]   FAR		- number of false alarms per frame
% [4]   GT        - number of ground truth trajectories
% [5-7] MT, PT, ML	- number of mostly tracked, partially tracked and mostly lost trajectories
% [8]   falsepositives- number of false positives (FP)
% [9]   missed        - number of missed targets (FN)
% [10]  idswitches	- number of id switches     (IDs)
% [11]  FRA       - number of fragmentations
% [12]  MOTA	- Multi-object tracking accuracy in [0,100]
% [13]  MOTP	- Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
% [14]  MOTAL	- Multi-object tracking accuracy in [0,100] with log10(idswitches)
%
% 
% (C) Anton Andriyenko, 2012
%
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without written permission from the authors.


% default options: 2D
if nargin<3
    options.eval3d=0;   % only bounding box overlap
    options.td=.5;      % threshold 50%
end

if ~isfield(options,'td')
    if options.eval3d
        options.td=1000;
    else
        options.td=0.5;
    end
end

td=options.td;

assert(length(gtInfo.frameNums)==length(stateInfo.frameNums), ...
    'Ground Truth and state must be of equal length');

assert(all(gtInfo.frameNums==stateInfo.frameNums), ...
    'Ground Truth and state must contain equal frame numbers');

% check if all necessery info is available
if options.eval3d
    assert(all(isfield(gtInfo,{'Xgp','Ygp'})), ...
        'Ground Truth Ground Plane coordinates needed for 3D evaluation');
    assert(all(isfield(stateInfo,{'Xgp','Ygp'})), ...
        'Ground Plane coordinates needed for 3D evaluation');
else
    assert(all(isfield(gtInfo,{'X','Y','W','H'})), ...
        'Ground Truth coordinates X,Y,W,H needed for 2D evaluation');
    assert(all(isfield(stateInfo,{'Xi','Yi','W','H'})), ...
        'State coordinates Xi,Yi,W,H needed for 2D evaluation');
    
end

gtInd=~~gtInfo.X;
stInd=~~stateInfo.X;

[Fgt, Ngt]=size(gtInfo.X);
[F, N]=size(stateInfo.X);

aspectRatio=mean(gtInfo.W(~~gtInfo.W)./gtInfo.H(~~gtInfo.H));
% gtInfo.W=gtInfo.H*aspectRatio;


metricsInfo.names.long = {'Recall','Precision','False Alarm Rate', ...
    'GT Tracks','Mostly Tracked','Partially Tracked','Mostly Lost', ...
    'False Positives', 'False Negatives', 'ID Switches', 'Fragmentations', ...
    'MOTA','MOTP', 'MOTA Log'};

metricsInfo.names.short = {'Rcll','Prcn','FAR', ...
    'GT','MT','PT','ML', ...
    'FP', 'FN', 'IDs', 'FM', ...
    'MOTA','MOTP', 'MOTAL'};

metricsInfo.widths.long = [6 9 16 9 14 17 11 15 15 11 14 5 5 8];
metricsInfo.widths.short = [5 5 5 3 3 3 3 4 4 3 3 5 5 5];

metricsInfo.format.long = {'.1f','.1f','.2f', ...
    'i','i','i','i', ...
    'i','i','i','i', ...
    '.1f','.1f','.1f'};

metricsInfo.format.short=metricsInfo.format.long;


metrics=zeros(1,14);
metrics(9)=numel(find(gtInd));  % False Negatives (missed)
metrics(7)=Ngt;                 % Mostly Lost

% nothing to be done, if state is empty
if ~N, return; end

% global opt
% if options.eval3d && opt.mex
%     [MOTA MOTP ma fpa mmea idsw missed falsepositives idswitches at afp MT PT ML rc pc faf FM MOTAL alld]= ...
%         CLEAR_MOT_mex(gtInfo.Xgp', gtInfo.Ygp', stateInfo.Xgp', stateInfo.Ygp',options.td);
% 
% %     cd /home/aanton/diss/utils
% %     [MOTA MOTP ma fpa mmea idsw missed falsepositives idswitches at afp MT PT ML rc pc faf FM MOTAL alld]= ...
% %         CLEAR_MOT(gtInfo.Xgp, gtInfo.Ygp, stateInfo.Xgp, stateInfo.Ygp,options.td);
% %     cd /home/aanton/visinf/projects/ongoing/contracking
%     metrics=[rc*100, pc*100, faf, Ngt, MT, PT, ML, falsepositives, missed, idswitches, FM, MOTA*100, MOTP*100, MOTAL*100];
%     metrics
%     global gsi
%     gsi=stateInfo;
%     pause
%     return;
% end


% mapping
M=zeros(F,Ngt);

mme=zeros(1,F); % ID Switchtes (mismatches)
c=zeros(1,F);   % matches found
fp=zeros(1,F);  % false positives
m=zeros(1,F);   % misses = false negatives
g=zeros(1,F);
d=zeros(F,Ngt);  % all distances;
ious=Inf*ones(F,Ngt);  % all overlaps

matched=@matched2d;
if options.eval3d, matched=@matched3d; end

alltracked=zeros(F,Ngt);
allfalsepos=zeros(F,N);

for t=1:F
    g(t)=numel(find(gtInd(t,:)));
    
    % mapping for current frame
    if t>1
        mappings=find(M(t-1,:));
        for map=mappings
            if gtInd(t,map) && stInd(t,M(t-1,map)) && matched(gtInfo,stateInfo,t,map,M(t-1,map),td)
                M(t,map)=M(t-1,map);
            end
        end
    end
    
    GTsNotMapped=find(~M(t,:) & gtInd(t,:));
    EsNotMapped=setdiff(find(stInd(t,:)),M(t,:));
    
    if options.eval3d
        alldist=Inf*ones(Ngt,N);
    
        mindist=0;
        while mindist < td && numel(GTsNotMapped)>0 && numel(EsNotMapped)>0
            for o=GTsNotMapped
                GT=[gtInfo.Xgp(t,o) gtInfo.Ygp(t,o)];
                for e=EsNotMapped
                    E=[stateInfo.Xgp(t,e) stateInfo.Ygp(t,e)];
                    alldist(o,e)=norm(GT-E);
                end
            end
            [mindist cind]=min(alldist(:));

            if mindist <= td
                [u v]=ind2sub(size(alldist),cind);
                M(t,u)=v;
                alldist(:,v)=Inf;
                GTsNotMapped=find(~M(t,:) & gtInd(t,:));
                EsNotMapped=setdiff(find(stInd(t,:)),M(t,:));
            end
        end
    
    else
        allisects=zeros(Ngt,N);        maxisect=Inf;
        
        while maxisect > td && numel(GTsNotMapped)>0 && numel(EsNotMapped)>0
            for o=GTsNotMapped
                GT=[gtInfo.X(t,o)-gtInfo.W(t,o)/2 ...
                    gtInfo.Y(t,o)-gtInfo.H(t,o) ...
                    gtInfo.W(t,o) gtInfo.H(t,o) ];
                for e=EsNotMapped
                    E=[stateInfo.Xi(t,e)-stateInfo.W(t,e)/2 ...
                        stateInfo.Yi(t,e)-stateInfo.H(t,e) ...
                        stateInfo.W(t,e) stateInfo.H(t,e) ];
                    allisects(o,e)=boxiou3D(GT(1),GT(2),GT(3),GT(4),E(1),E(2),E(3),E(4));
                end
            end
            [maxisect cind]=max(allisects(:));

            if maxisect >= td
                [u v]=ind2sub(size(allisects),cind);
                M(t,u)=v;
                allisects(:,v)=0;
                GTsNotMapped=find(~M(t,:) & gtInd(t,:));
                EsNotMapped=setdiff(find(stInd(t,:)),M(t,:));
            end

        end
    end
    
    curtracked=find(M(t,:));
    
    
    alltrackers=find(stInd(t,:));
    mappedtrackers=intersect(M(t,find(M(t,:))),alltrackers);
    falsepositives=setdiff(alltrackers,mappedtrackers);
    
    alltracked(t,:)=M(t,:);
    allfalsepos(t,1:length(falsepositives))=falsepositives;
    
    %%  mismatch errors
    if t>1
        for ct=curtracked
            lastnotempty=find(M(1:t-1,ct),1,'last');
            if gtInd(t-1,ct) && ~isempty(lastnotempty) && M(t,ct)~=M(lastnotempty,ct)
                mme(t)=mme(t)+1;
            end
        end
    end
    
    c(t)=numel(curtracked);
    for ct=curtracked
        eid=M(t,ct);
        if options.eval3d
            d(t,ct)=norm([gtInfo.Xgp(t,ct) gtInfo.Ygp(t,ct)] - ...
                [stateInfo.Xgp(t,eid) stateInfo.Ygp(t,eid)]);
        else
            gtLeft=gtInfo.X(t,ct)-gtInfo.W(t,ct)/2;
            gtTop=gtInfo.Y(t,ct)-gtInfo.H(t,ct);
            gtWidth=gtInfo.W(t,ct);    gtHeight=gtInfo.H(t,ct);
            
            stLeft=stateInfo.Xi(t,eid)-stateInfo.W(t,eid)/2;
            stTop=stateInfo.Yi(t,eid)-stateInfo.H(t,eid);
            stWidth=stateInfo.W(t,eid);    stHeight=stateInfo.H(t,eid);
            ious(t,ct)=boxiou3D(gtLeft,gtTop,gtWidth,gtHeight,stLeft,stTop,stWidth,stHeight);
        end
    end
    
    
    fp(t)=numel(find(stInd(t,:)))-c(t);
    m(t)=g(t)-c(t);
    
    
end    

missed=sum(m);
falsepositives=sum(fp);
idswitches=sum(mme);

if options.eval3d
    MOTP=(1-sum(sum(d))/sum(c)/td) * 100; % avg distance to [0,100]
else
    MOTP=sum(ious(ious>=td & ious<Inf))/sum(c) * 100; % avg ol
end

MOTAL=(1-((sum(m)+sum(fp)+log10(sum(mme)+1))/sum(g)))*100;
MOTA=(1-((sum(m)+sum(fp)+(sum(mme)))/sum(g)))*100;
recall=sum(c)/sum(g)*100;
precision=sum(c)/(sum(fp)+sum(c))*100;
FAR=sum(fp)/Fgt;
 

%% MT PT ML
MTstatsa=zeros(1,Ngt);
for i=1:Ngt
    gtframes=find(gtInd(:,i));
    gtlength=length(gtframes);
    gttotallength=numel(find(gtInd(:,i)));
    trlengtha=numel(find(alltracked(gtframes,i)>0));
    if gtlength/gttotallength >= 0.8 && trlengtha/gttotallength < 0.2
        MTstatsa(i)=3;
    elseif t>=find(gtInd(:,i),1,'last') && trlengtha/gttotallength <= 0.8
        MTstatsa(i)=2;
    elseif trlengtha/gttotallength >= 0.8
        MTstatsa(i)=1;
    end
end
% MTstatsa
MT=numel(find(MTstatsa==1));PT=numel(find(MTstatsa==2));ML=numel(find(MTstatsa==3));

%% fragments
fr=zeros(1,Ngt);
for i=1:Ngt
    b=alltracked(find(alltracked(:,i),1,'first'):find(alltracked(:,i),1,'last'),i);
    b(~~b)=1;
    fr(i)=numel(find(diff(b)==-1));
end
FRA=sum(fr);

assert(Ngt==MT+PT+ML,'Hmm... Not all tracks classified correctly.');
metrics=[recall, precision, FAR, Ngt, MT, PT, ML, falsepositives, missed, idswitches, FRA, MOTA, MOTP, MOTAL];
end 




function ret=matched2d(gtInfo,stateInfo,t,map,mID,td)
    gtLeft=gtInfo.X(t,map)-gtInfo.W(t,map)/2;
    gtTop=gtInfo.Y(t,map)-gtInfo.H(t,map);
    gtWidth=gtInfo.W(t,map);    gtHeight=gtInfo.H(t,map);
    
    stLeft=stateInfo.Xi(t,mID)-stateInfo.W(t,mID)/2;
    stTop=stateInfo.Yi(t,mID)-stateInfo.H(t,mID);
    stWidth=stateInfo.W(t,mID);    stHeight=stateInfo.H(t,mID);
    
    ret = boxiou3D(gtLeft,gtTop,gtWidth,gtHeight,stLeft,stTop,stWidth,stHeight) >= td;    
end


function ret=matched3d(gtInfo,stateInfo,t,map,mID,td)
    Xgt=gtInfo.Xgp(t,map); Ygt=gtInfo.Ygp(t,map);
    X=stateInfo.Xgp(t,mID); Y=stateInfo.Ygp(t,mID);
    ret=norm([Xgt Ygt]-[X Y])<=td;

end