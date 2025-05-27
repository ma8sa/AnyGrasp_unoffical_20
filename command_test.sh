#CUDA_VISIBLE_DEVICES=4 python test.py --camera kinect --dump_dir ./ --checkpoint_path minkuresunet_realsense.tar --batch_size 1 --dataset_root /data3/graspnet --infer --eval --collision_thresh -1
python test.py --camera realsense --dump_dir ./ --checkpoint_path minkuresunet_realsense.tar --batch_size 1 --dataset_root ./dataset/data/ --infer --eval --collision_thresh -1
