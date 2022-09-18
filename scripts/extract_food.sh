
JSON_PATH=/data/full-version-vip-pro/annotations/$1.json
OUT_FILE='.cache/out_jsons/pred_food_on_'$1'.json'

IMG_DIR="DMS_DB_090922"
JSON_TEST="/data/DMS_Behavior_Detection/merge-phone-cigaret-food/annotations/val.json"


cmd="python tools/eval_export.py -f exps/dms/food_m.py --json_path $JSON_PATH -d 1 \
    --json_test $JSON_TEST --out_file $OUT_FILE val_name $IMG_DIR"
echo $cmd
$cmd
