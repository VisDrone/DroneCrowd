function  sorttrackData = classSplit(trackData)
% object category:
%     ignored |pedestrian| person| bicycle| car| van| truck| tricycle| awning-tricycle  | bus| motor| others
%        0    |    1     |  2    |  3     | 4  | 5  | 6    |  7      |        8         | 9  | 10   |  11  

sorttrackData.pedestrian = trackData(trackData(:,8)==1,:);
sorttrackData.person = trackData(trackData(:,8)==2,:);
sorttrackData.bicycle = trackData(trackData(:,8)==3,:);
sorttrackData.car = trackData(trackData(:,8)==4,:);
sorttrackData.van = trackData(trackData(:,8)==5,:);
sorttrackData.truck = trackData(trackData(:,8)==6,:);
sorttrackData.tricycle = trackData(trackData(:,8)==7,:);
sorttrackData.awningtricycle = trackData(trackData(:,8)==8,:);
sorttrackData.bus = trackData(trackData(:,8)==9,:);
sorttrackData.motor = trackData(trackData(:,8)==10,:);