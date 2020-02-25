python3 -W ignore train.py --model_def config/yolov3-custom_human.cfg --data_config config/dataset.data --pretrained_weights weights/yolov3.weights --batch_size=10 --epochs 150
#python3 -W ignore train.py --model_def config/yolov3-custom_human.cfg --data_config config/dataset.data --pretrained_weights checkpoints/yolov3_ckpt_99.pth --batch_size=10 --epochs 50

retval=$?

if [ $retval -eq 0 ]; then
    echo -e "\e[1;32m Deleting checkpoints... \e[0m"
    rm checkpoints/yolov3_ckpt_0*
    rm checkpoints/yolov3_ckpt_2*
    rm checkpoints/yolov3_ckpt_3*
    rm checkpoints/yolov3_ckpt_4*
    rm checkpoints/yolov3_ckpt_5*
    rm checkpoints/yolov3_ckpt_6*
    rm checkpoints/yolov3_ckpt_7*
    rm checkpoints/yolov3_ckpt_8*
    #rm checkpoints/yolov3_ckpt_9*
    #rm checkpoints/yolov3_ckpt_10*
else
    echo -e "\e[1;31m Training aborted! \e[0m"
fi


#spd-say 'Training finished'
