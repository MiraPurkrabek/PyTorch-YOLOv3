# Detect test images by model 99
python3 new_detect.py --image_folder data/dataset/test/ --checkpoint_model checkpoints/yolov3_ckpt_99.pth --weights_path checkpoints/yolov3_ckpt_99.pth --model_def config/yolov3-custom.cfg --class_path data/dataset/classes.names
