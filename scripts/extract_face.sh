
JSON_PATH=/data/DMS_Behavior_Detection/merge-phone-cigaret-food/annotations/$1.json
OUT_FILE='.cache/out_jsons/pred_face_on_'$1'.json'

IMG_DIR=images

CKPT=YOLOX_outputs/face_m/best_ckpt.pth
JSON_TEST=/data/full-version-vip-pro/annotations/val.json


cmd="python tools/eval_export.py -f exps/dms/face_m.py --json_path $JSON_PATH -d 1 \
    --json_test $JSON_TEST --out_file $OUT_FILE val_name $IMG_DIR"
echo $cmd
$cmd