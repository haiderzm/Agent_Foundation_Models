set -x

ulimit -n 65535
# =====================================================================================================================
#                                      Param
# =====================================================================================================================
ACTOR_LR=1e-6

TRAIN_BS=3
PPO_MINI_BS=3
GEN_BS=3

EPOCHS=3
STEPS=320
# STEPS=150
N=4

PPO_MICRO_BSZ_PER_GPU=1
LOG_PROB_MICRO_BSZ_PER_GPU=1

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28

# context window
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 14))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

# performance related param
SP_SIZE=1
GEN_TP=1
use_dynamic_bsz=True
offload=True

# =====================================================================================================================
#                                      Env
# =====================================================================================================================
CURRENT_DIR=$(pwd)
export CUDA_VISIBLE_DEVICES=5,6,7
export RAY_BACKEND_LOG_LEVEL=debug
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG_SUBSYS=ALL
export NNODES=1
export PROJECT_NAME="qwen3_8b_base_rl"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_DISABLE_FOREACH=1

EXPERIMENT_DIR=$(dirname "$(readlink -f "$0")")

export WANDB_MODE="online"
export WANDB_PROJECT="qwen3_8b_base_rl"
export EXPERIMENT_NAME="QWEN3-VL-8B-SFT-N-4-sglang-BS-3-Steps-320-Epoch-3-C-16k"
# export BASE_MODEL="/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen3-8B"   # your train model path
export BASE_MODEL="/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/models/Qwen3-8B-sft-CoA-1828"   # your train model path
export VLLM_ATTENTION_BACKEND=XFORMERS

CKPT_DIR="/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/AFM/train/web_agent/rl/QWEN3-VL-8B-SFT-N-4-sglang-BS-3-Steps-320-Epoch-3-C-16k/global_step_150"

TRAIN_DATASETS="/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/AFM-WebAgent-RL-Dataset/train_split.parquet"   # your train dataset
VAL_DATASETS="/home/Md.Zama@mbzuai.ac.ae/Agent_Foundation_Models/AFM-WebAgent-RL-Dataset/val_split.parquet" # "your val datasets"

# =====================================================================================================================
#                                      Tool
# =====================================================================================================================
# code tool
CODE_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/code_tool_config/code_executor.yaml"
# search tools
SEARCH_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/search_tool_config/training_servers_config.yaml"
# afm tools
AFM_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/afm_tool_config/afm_tool_config.yaml" 

# =====================================================================================================================
#                                      Train
# =====================================================================================================================
cd verl
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.resume_mode="resume_path" \
    trainer.resume_from_path="$CKPT_DIR" \
    algorithm.filter_groups.enable=true \
    data.train_files=[\"${TRAIN_DATASETS}\"] \
    data.val_files=[\"${VAL_DATASETS}\"] \
    data.train_batch_size="${TRAIN_BS}" \
    +data.gen_batch_size="${GEN_BS}" \
    data.val_batch_size=256 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.shuffle=true \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=3 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.checkpoint.save_contents="['model', 'optimizer', 'extra']" \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.max_model_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    trainer.logger=['wandb','tensorboard'] \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=50 \
    trainer.test_freq=1000 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs="${EPOCHS}" \
    trainer.total_training_steps="${STEPS}" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${EXPERIMENT_DIR}/${EXPERIMENT_NAME}" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_turns=25 \
    +actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.use_xml_tool_parser=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$SEARCH_CONFIG" \
    reward_model.reward_manager="batch" \
    custom_reward_function.train_path="${CURRENT_DIR}/verl/verl/utils/reward_score/grm_simple.py" \
    custom_reward_function.train_name="compute_score_grm_batch" \
    custom_reward_function.val_path="${CURRENT_DIR}/verl/verl/utils/reward_score/grm_simple.py" \
    custom_reward_function.val_name="compute_score_grm_batch" \
    +actor_rollout_ref.ref.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    2>&1 | tee $EXPERIMENT_DIR/$EXPERIMENT_NAME.log
    # actor_rollout_ref.actor.use_torch_compile=false \
    # actor_rollout_ref.ref.use_torch_compile=false \
    # actor_rollout_ref.rollout.enable_chunked_prefill=false \