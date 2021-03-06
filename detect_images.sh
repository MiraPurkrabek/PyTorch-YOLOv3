# Detect US images by model 99
#python3 detect.py --image_folder data/dataset/unannotated_sequences/ --checkpoint_model checkpoints/yolov3_ckpt_99.pth --weights_path checkpoints/yolov3_ckpt_99.pth --model_def config/yolov3-custom.cfg --class_path data/dataset/classes.names

# Detect test images by model 99
#python3 detect.py --image_folder data/dataset/test/ --checkpoint_model checkpoints/yolov3_ckpt_99.pth --weights_path checkpoints/yolov3_ckpt_99.pth --model_def config/yolov3-custom.cfg --class_path data/dataset/classes.names

# Detect test images by model sq_200
#python3 detect.py --image_folder data/THA-AUS/ --checkpoint_model checkpoints/yolov3_ckpt_sq_200.pth --weights_path checkpoints/yolov3_ckpt_sq_200.pth --model_def config/yolov3-custom.cfg --class_path data/dataset/classes.names

# Detect test images by model sq_200
python3 detect.py --image_folder data/dataset/test --checkpoint_model checkpoints/yolov3_ckpt_145.pth --weights_path checkpoints/yolov3_ckpt_145.pth --model_def config/yolov3-custom_human.cfg --class_path data/dataset/classes.names

# Detect test images by model sq_50
#python3 detect.py --image_folder data/dataset/test/ --checkpoint_model checkpoints/yolov3_ckpt_sq_50.pth --weights_path checkpoints/yolov3_ckpt_sq_50.pth --model_def config/yolov3-custom.cfg --class_path data/dataset/classes.names

#spd-say 'Images detected'