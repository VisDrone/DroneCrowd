function tendMets = classEval(gtsortdata, ressortdata, allMets, ind, evalClassSet, sequenceName)

threshold = 0.5;
world = 0;

for k = 1:length(evalClassSet)
    className = evalClassSet{k};
    classID = getClassID(className);
    [gtdata, resdata] = selectTrackData(gtsortdata, ressortdata, classID);
    if(~isempty(gtdata))
        [metsCLEAR, ~, additionalInfo] = CLEAR_MOT_HUN(gtdata, resdata, threshold, world);
        metsID = IDmeasures(gtdata, resdata, threshold, world);
        mets = [metsID.IDF1, metsID.IDP, metsID.IDR, metsCLEAR];
        allMets(ind).name = strcat(sequenceName, '(', className, ')');
        allMets(ind).m    = mets;
        allMets(ind).IDmeasures = metsID;
        allMets(ind).additionalInfo = additionalInfo;
        tendMets(k) = allMets(ind);
    else
        allMets(ind).name = strcat(sequenceName, '(', className, ')');
        allMets(ind).m    = [];
        allMets(ind).IDmeasures = [];
        allMets(ind).additionalInfo = [];
        tendMets(k) = allMets(ind);        
    end
end