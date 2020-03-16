WEIGHTS='yolov3.weights'
CONFIG='config/custom.data'
CFG='yolov3-custom.cfg'
#CFG='yolov3-twoheads.cfg'

python3 -W ignore train.py --model_def config/$CFG --data_config $CONFIG --pretrained_weights weights/$WEIGHTS --batch_size=10

#spd-say 'Training finished'
