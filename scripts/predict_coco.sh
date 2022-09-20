%load_ext autoreload
%autoreload 2
%run tools/predict_coco.py -f exps/dms/mb2_face_food.py -c YOLOX_outputs/mb2_face_food/best_ckpt.pth \
    -i .cache/raw_video_predict_face_food/aaron_mobile_usage_0173/annotations/pred_mb2_face_food.json  -o ff.json -d 1


# JSON_PATH=$1
# OUT_JSON_PATH=$2
# python tools/predict_coco.py -f exps/dms/mb2_face_food.py -c YOLOX_outputs/mb2_face_food/best_ckpt.pth \
#     -i $JSON_PATH  -o $OUT_JSON_PATH -d 1 -b 16

