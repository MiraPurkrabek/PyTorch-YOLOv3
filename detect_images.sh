# MODEL='yolov3.cfg'
MODEL='yolov3_human_only_no_anchors_200.pth'
# MODEL='yolov3_ckpt_99.pth'
# MODEL='yolov3_human_multibox_noNMS.pth'
# DATA='data/dataset/test'
# DATA='data/dataset/unannotated_sequences/US1'
# DATA='data/dataset/unannotated_sequences/US2'
# DATA='data/det_bench'
DATA='data/AS22'
# DATA='data/THA-AUS/selection/'
# DATA='data/dataset/hockey/'
# CFG='yolov3.cfg'
# CFG='yolov3-custom.cfg'
CFG='yolov3-anchors_200.cfg'
# CFG='yolov3-4class.cfg'
# CFG='yolov3-twoheads.cfg'

rm -r output/*
python3 detect.py --image_folder $DATA --checkpoint_model checkpoints/$MODEL --weights_path checkpoints/$MODEL --model_def config/$CFG --class_path data/dataset/classes.names
# python3 detect.py --image_folder $DATA --checkpoint_model weights/yolov3.weights --weights_path weights/yolov3.weights --model_def config/$CFG --class_path data/dataset/classes.names

#spd-say 'Images detected'