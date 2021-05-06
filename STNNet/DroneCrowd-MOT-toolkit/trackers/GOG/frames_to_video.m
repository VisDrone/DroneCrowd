% converts bunch of frames to a video file
function frames_to_video(frames_path, video_fname, frame_rate)
if ~exist('frame_rate')
  frame_rate = 5;
end
% unix(['ffmpeg -y -i ' frames_path '%6d.jpg -ar 22050 -b 50000 -vtag DIVX
% -f avi ' video_fname]);
% unix(['ffmpeg -y -r ' num2str(frame_rate) ' -i ' frames_path '%6d.jpg -ar 22050 -b 50000 -r 24 -vtag DIVX -f avi ' video_fname]);
unix(['ffmpeg -y -r ' num2str(frame_rate) ' -i ' frames_path '%8d.jpg -b 1000000 -vtag DIVX -f avi ' video_fname]);