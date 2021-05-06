function showResults(isSeqDisplay, res, sequence)

if(isSeqDisplay)
    sceneInfo.frames = sequence.dataset;
    sceneInfo.imgHeight = sequence.imgHeight;
    sceneInfo.imgWidth = sequence.imgWidth;
    sceneInfo.frameNums = 1:length(sequence.dataset);
    
    %% Display parameters
    sceneInfo.defaultColor = [.1 .2 .9];
    sceneInfo.grey = 0.7*ones(1,3);

    sceneInfo.traceLength = 20; % overlay track from past n frames
    sceneInfo.dotSize = 15;
    sceneInfo.boxLineWidth = 1;
    sceneInfo.traceWidth = 0.5;

    % what to display
    sceneInfo.displayDots = true; % display the trajectory dots
    sceneInfo.displayBoxes = false; % display the bounding box of the target
    sceneInfo.displayID = false; % display the ID of the target
    sceneInfo.displayCropouts = false; % display the border cropouts
    sceneInfo.displayConnections = false; % display the connection of the sample target    

    displayTrackingResult(sceneInfo, res, sequence.seqName);
end