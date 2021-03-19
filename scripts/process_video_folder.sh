#!/bin/sh
# sh scripts/process_video_folder.sh <config> <video-directory> [additional parameter]
for video in $2/*.mp4;
do
  echo $video;
  python ufld.py $1 --mode=runtime --input_mode video --video_input_file "$video" --output_mode video --video_out_enable_live_video False --video_out_enable_video_export True $3;
done