### ----------------- COCO Base Training Part ------------------------- ###
rm support_dir/support_feature.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 fsod_train_net.py --num-gpus 8 \
	--config-file configs/fsod/R101/R_101_C4_1x.yaml 2>&1 | tee log/fsod_train_log.txt
### Fine-tuning
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 fsod_train_net.py --num-gpus 8 \
	--config-file configs/fsod/R101/FT_R_101_C4_1x.yaml 2>&1 | tee log/fsod_train_log.txt