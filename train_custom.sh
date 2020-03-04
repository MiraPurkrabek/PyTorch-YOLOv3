CONFIG='config/custom.data'
CFG='yolov3-twoheads.cfg'

python3 -W ignore train.py --model_def config/$CFG --data_config $CONFIG  --epochs 50 --batch_size 10
