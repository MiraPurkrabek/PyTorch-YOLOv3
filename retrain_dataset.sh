# CFG='yolov3-custom.cfg'
# CFG='yolov3-anchors_150.cfg'

CFG='yolov3-4class.cfg'

python3 -W ignore train.py --model_def config/$CFG --data_config config/dataset.data --pretrained_weights weights/yolov3.weights --batch_size=10

retval=$?

if [ $retval -eq 0 ]; then
    echo -e "\e[1;32m Deleting checkpoints... \e[0m"
    rm checkpoints/yolov3_ckpt_0*
    rm checkpoints/yolov3_ckpt_1*
    rm checkpoints/yolov3_ckpt_2*
    rm checkpoints/yolov3_ckpt_3*
    rm checkpoints/yolov3_ckpt_4*
    rm checkpoints/yolov3_ckpt_5*
    rm checkpoints/yolov3_ckpt_6*
    rm checkpoints/yolov3_ckpt_7*
    rm checkpoints/yolov3_ckpt_8*
else
    echo -e "\e[1;31m Training aborted! \e[0m"
fi


#spd-say 'Training finished'
