function [gtdata, resdata] = selectTrackData(gtsortdata, ressortdata, classID)

% object category:
%     ignored |pedestrian| person| bicycle| car| van| truck| tricycle| awning-tricycle  | bus| motor| others
%        0    |    1     |  2    |  3     | 4  | 5  | 6    |  7      |        8         | 9  | 10   |  11  

switch(classID)
    case 1
        gtdata = gtsortdata.pedestrian;
        resdata = ressortdata.pedestrian;
    case 2
        gtdata = gtsortdata.person;
        resdata = ressortdata.person;     
    case 3
        gtdata = gtsortdata.bicycle;
        resdata = ressortdata.bicycle;
    case 4
        gtdata = gtsortdata.car;
        resdata = ressortdata.car;
    case 5
        gtdata = gtsortdata.van;
        resdata = ressortdata.van;
    case 6
        gtdata = gtsortdata.truck;
        resdata = ressortdata.truck;
    case 7
        gtdata = gtsortdata.tricycle;
        resdata = ressortdata.tricycle;           
    case 8
        gtdata = gtsortdata.awningtricycle;
        resdata = ressortdata.awningtricycle;   
    case 9
        gtdata = gtsortdata.bus;
        resdata = ressortdata.bus;   
    case 10
        gtdata = gtsortdata.motor;
        resdata = ressortdata.motor;           
end