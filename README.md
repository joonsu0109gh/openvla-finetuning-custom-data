# OpenVLA Finetuning with My Own Data

This guide outlines the steps to finetune the OpenVLA pre-trained model using your own dataset.

## Prerequisites
- Set up the environment by following the [OpenVLA repository](https://github.com/openvla/openvla).

## Finetuning Steps

1. **Prepare Your Dataset**
   - Copy your dataset to the `./openvla` directory.

2. **Create Checkpoint Directory**
   - Create a directory to save model checkpoints:
     ```shell
     mkdir -p /home/rvi/openvla/ckpts/ckpt_finetune_5episode_same_position
     ```
    - Refer this [repository](https://github.com/joonsu0109gh/franka_data_collecting) for data collection.

3. **Finetune the Model**
   - Run one of the following commands to finetune the OpenVLA model. Adjust the options (e.g., paths, dataset name, hyperparameters) to match your setup.
   
   **Option 1: Using `torchrun`**
   ```shell
   torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
     --vla_path "openvla/openvla-7b" \
     --data_root_dir {e.g., /home/rvi/openvla/dataset} \
     --dataset_name {e.g., my_dataset} \
     --run_root_dir {e.g., /home/rvi/openvla/ckpts/ckpt_finetune_5episode_same_position} \
     --adapter_tmp_dir {e.g., /home/rvi/openvla/test_log} \
     --lora_rank 32 \
     --batch_size 1 \
     --grad_accumulation_steps 1 \
     --learning_rate 5e-4 \
     --image_aug False \
     --wandb_project openvla \
     --wandb_entity {} \
     --save_steps 2000 \
     --max_steps 20000 \
     --save_latest_checkpoint_only False
   ```

   **Option 2: Using `torch.distributed.run`**
   ```shell
   /home/rvi/miniconda3/envs/openvla/bin/python -m torch.distributed.run --standalone --nproc-per-node 1 \
     vla-scripts/finetune.py \
     --vla_path "openvla/openvla-7b" \
     --data_root_dir {e.g., /home/rvi/openvla/dataset} \
     --dataset_name {e.g., my_dataset} \
     --run_root_dir {e.g., /home/rvi/openvla/ckpts/ckpt_finetune_5episode_same_position} \
     --adapter_tmp_dir {e.g., /home/rvi/openvla/test_log} \
     --lora_rank 32 \
     --batch_size 1 \
     --grad_accumulation_steps 1 \
     --learning_rate 5e-4 \
     --image_aug False \
     --wandb_project openvla \
     --wandb_entity {} \
     --save_steps 2000 \
     --max_steps 20000 \
     --save_latest_checkpoint_only False
   ```

## Deployment
- After finetuning, deploy the model using the following command. Replace `{path/to/checkpoint}` with the path to your finetuned checkpoint and `{task instruction}` with the desired task instruction:
  ```shell
  sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH $(which python) my_test_pretrained.py \
    --checkpoint-path {path/to/checkpoint} \
    --instruction "{task instruction}"
  ```
