### ----------------- COCO Testing Part ------------------------- ###
for shots in 3
do
	# generate few-shot support
	rm support_dir/support_feature.pkl
	CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
		--config-file configs/fsod/R101/test_R_101_C4_1x_subt${shots}_a.yaml \
		--eval-only 2>&1 | tee log/fsod_101_test_log_subt${shots}_a.txt
    --opts 
	# evaluation
	CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
		--config-file configs/fsod/R101/test_R_101_C4_1x_subt${shots}_a.yaml \
		--eval-only 2>&1 | tee log/fsod_101_test_log_subt${shots}_a.txt
done
