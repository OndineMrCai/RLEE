#!/bin/bash
#SBATCH --job-name=rlee-ray-on-slurm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --account=def-hongyanz
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --exclude=ng30811,ng30805
#SBATCH --output=/home/sccai/projects/def-hongyanz/sccai/reasoning/RLEE/logs/rlee_deepscaler-%j.out
#SBATCH --error=/home/sccai/projects/def-hongyanz/sccai/reasoning/RLEE/logs/rlee_deepscaler-%j.err

echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "nodes_array=(${nodes_array[@]})"
echo "worker_num=$worker_num"

# replace these with your paths
rlee_workdir=/home/sccai/projects/def-hongyanz/sccai/RLEE
MODEL_PATH="/scratch/sccai/reasoning_model/DeepScaleR-1.5B-Preview"
savepath="/home/sccai/projects/def-hongyanz/sccai/reasoning/RLEE/ckpt"
datapath="./rlee/data"
reward_type=rlee
# replace these with your paths

# Get the list of nodes
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Convert to IPv4 if necessary
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "
  source rain/bin/activate
  cd $rlee_workdir
  ray start --head --node-ip-address=$head_node_ip --port=$port \
      --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus ${SLURM_GPUS_PER_NODE} --block
" &

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" bash -c "
      source rain/bin/activate
      cd $rlee_workdir
      ray start --address $ip_head --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus ${SLURM_GPUS_PER_NODE} --block
    " &
    sleep 5
done

# Start training script on the head node
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" bash -c "
  source rain/bin/activate
  cd $rlee_workdir
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$datapath/train/train.parquet \
    data.val_files=[$datapath/test/aime.parquet,$datapath/test/amc.parquet,$datapath/test/math.parquet] \
    data.train_batch_size=64 \
    data.val_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='RLEE' \
    trainer.experiment_name='main' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=2 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=$savepath \
    trainer.reward_type=$reward_type
" 2>&1 | tee verl_demo_slurm.log
