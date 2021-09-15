python tools/demo.py image -n yolox-m -c weights/pretrained/yolox_m.pth \
    --path datasets/tsd/test/test_data_20210524/test_images_5k/ --conf 0.025 --nms 0.45 --tsize 640 --save_result --device gpu

python tools/demo.py image -n yolox-m -c weights/pretrained/yolox_m.pth \
    --path assets/dog.jpg --conf 0.025 --nms 0.45 --tsize 640 --save_result --device gpu

    