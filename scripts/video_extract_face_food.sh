VIDEO=$2
CUDA_VISIBLE_DEVICES=$1 python tools/eval_export.py -f exps/dms/mb2_face_food.py --input_video $VIDEO  --out_dir .cache/raw_video_predict_face_food -d 1

#/data/DMS_Behavior_Detection/RawVideos/Action_Eating/*/*.mp4
