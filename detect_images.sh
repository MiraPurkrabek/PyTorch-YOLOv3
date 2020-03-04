MODEL='yolov3_ckpt_99.pth'
DATA='data/dataset/test'
#DATA='data/dataset/unannotated_sequences/'
#DATA='data/THA-AUS/selection/'
#CFG='yolov3-custom.cfg'
CFG='yolov3-twoheads.cfg'

python3 detect.py --image_folder $DATA --checkpoint_model checkpoints/$MODEL --weights_path checkpoints/$MODEL --model_def config/$CFG --class_path data/dataset/classes.names

#spd-say 'Images detected'