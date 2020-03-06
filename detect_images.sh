MODEL='yolov3_human_multibox_noNMS.pth'
#DATA='data/dataset/test'
#DATA='data/dataset/unannotated_sequences/US1'
#DATA='data/THA-AUS/selection/'
DATA='data/dataset/hockey/'
CFG='yolov3-custom.cfg'
#CFG='yolov3-twoheads.cfg'

rm output/*
python3 detect.py --image_folder $DATA --checkpoint_model checkpoints/$MODEL --weights_path checkpoints/$MODEL --model_def config/$CFG --class_path data/dataset/classes.names

#spd-say 'Images detected'