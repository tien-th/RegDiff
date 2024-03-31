# train

# python3 main.py --config configs/25_03_Ldm_reg.yaml --train --sample_at_start --save_top --gpu_ids 1 
# --resume_model ./results/LDM_reg_smooth_03_0.15/LDM-f4/checkpoint/latest_model_156.pth --resume_optim ./results/LDM_reg_smooth_03_0.15/LDM-f4/checkpoint/latest_optim_sche_156.pth

#test
python3 main.py --config configs/25_03_Ldm_reg.yaml --sample_to_eval --gpu_ids 2 --resume_model ./results/LDM_reg_smooth_03_0.15/LDM-f4/checkpoint/latest_model_200.pth 


## rename
#python3 preprocess_and_evaluation.py -f rename_samples -r root/dir -s source/dir -t target/dir

## copy
#python3 preprocess_and_evaluation.py -f copy_samples -r root/dir -s source/dir -t target/dir

## LPIPS
#python3 preprocess_and_evaluation.py -f LPIPS -s source/dir -t target/dir -n 1

## max_min_LPIPS
#python3 preprocess_and_evaluation.py -f max_min_LPIPS -s source/dir -t target/dir -n 1

## diversity
#python3 preprocess_and_evaluation.py -f diversity -s source/dir -n 1

## fidelity
#fidelity --gpu 0 --fid --input1 path1 --input2 path2


# OMP_NUM_THREADS=12 
# -m torch.distributed.run --nnodes=1 --nproc_per_node=8 
# --sample_to_eval
# --resume_model results/LBBDMxSAGxVq13/LBBDM-f4/checkpoint/top_model_epoch_192.pth --resume_optim results/LBBDMxSAGxVq13/LBBDM-f4/checkpoint/top__optim_sche_epoch_192.pth 