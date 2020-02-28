# Detect test images by model sq_200
python3 detect.py --image_folder data/THA-AUS/selection/ --checkpoint_model checkpoints/yolov3_ckpt_99.pth --weights_path checkpoints/yolov3_ckpt_99.pth --model_def config/yolov3-custom.cfg --class_path data/dataset/classes.names
#python3 detect.py --image_folder data/dataset/test/ --checkpoint_model checkpoints/yolov3_ckpt_99.pth --weights_path checkpoints/yolov3_ckpt_99.pth --model_def config/yolov3-custom.cfg --class_path data/dataset/classes.names
#python3 detect.py --image_folder data/dataset/unannotated_sequences/ --checkpoint_model checkpoints/yolov3_ckpt_99.pth --weights_path checkpoints/yolov3_ckpt_99.pth --model_def config/yolov3-custom.cfg --class_path data/dataset/classes.names

#spd-say 'Images detected'