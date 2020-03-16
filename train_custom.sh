CONFIG='config/custom.data'
#CFG='yolov3-twoheads.cfg'
CFG='yolov3-custom.cfg'

python3 -W ignore train.py --model_def config/$CFG --data_config $CONFIG  --epochs 50 --batch_size 1
