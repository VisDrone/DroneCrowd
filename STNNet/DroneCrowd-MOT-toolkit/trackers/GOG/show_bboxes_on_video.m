% it shows bboxes on frames of a video (bunch of frames in "input_frames") and outputs a video "video_fname"
% bboxes(i).bbox is an n*5 matrix: detections on the i'th frame
% "thr" is used to prune detection results: default: -inf
% if you don't pass "output_frames", it will make a temporary one and then delete it in the end.
% "bws" is track numbers in image format (used as label for the boxes)

function show_bboxes_on_video(input_frames, bboxes, video_fname, bws, frame_rate, thr, output_frames, frameNums)

if ~exist('frame_rate', 'var')
  frame_rate = 20;
end
if ~exist('thr', 'var')
  thr = -inf;
end
if exist('output_frames', 'var')
  if ~isempty(output_frames)
    flag = 0;
%     unix(['rm -r ' output_frames]);
  else
    flag = 1;
  end
else
  flag = 1;
end
if flag == 1
  output_frames = tempname;   %% A temporary folder name
  output_frames = [output_frames(end-9:end) '/'];
end

% mkdir (output_frames);

col = round((rand(3,1e4)/2+.5)*255);  %% we assume number of tracks is less than 1e4.

dirlist = dir([input_frames '*.jpg']);  %%list of images
if isempty(dirlist)
  dirlist = dir([input_frames '*.png']);
end

figure(1), title('Tracking Results');
% [imgWidth, imgHeight] = size(imread(fullfile(input_frames, [sprintf('img%0.5d', frameNums(1)) '.jpg'])));
% figure('Position',[100 200 imgWidth imgHeight]), title('Tracking Results');
for i = 1:length(bboxes)
  bbox = bboxes(i).bbox;
%   img = imread(fullfile(input_frames, [sprintf('frame_%0.4d', frameNums(i)-1) '.jpg']));%% read an image    
  img = imread(fullfile(input_frames, [sprintf('img%0.5d', frameNums(i)) '.jpg']));%% read an image
  if ~isempty(bbox)
    img = show_bbox_on_image(img, bbox(bbox(:,end) > thr, :), bws, col);
  end
  imshow(img);
  % frame number
  text(20,50,sprintf('%d',i),'FontSize',20);  
  pause(0.01);  
%   imwrite(img, [output_frames sprintf('%0.4d', i) '.jpg']); %%write the output image
%   imwrite(img, [output_frames sprintf('%0.5d', i) '.jpg']); %%write the output image  
end

frames_to_video(output_frames, video_fname, frame_rate);  %%convert frames to video

if flag
%   unix(['rm -r ' output_frames]); %%remove temporary output folder
end
