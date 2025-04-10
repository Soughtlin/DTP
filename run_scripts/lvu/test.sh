checkpoint_path=$1
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    --master_port=34653 \
    train.py \
    --cfg-path lavis/projects/malmm/cls_lvu.yaml \
    --options \
    model.arch blip2_vicuna_instruct_dtp \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 20 \
    model.num_frames 100 \
    datasets.lvu_cls.history 100 \
    datasets.lvu_cls.task genre \
    datasets.lvu_cls.stride 20 \
    run.init_lr 1e-4 \
    run.max_epoch 20 \
    run.num_beams 5 \
    run.batch_size_train 32 \
    run.batch_size_eval 32 \
    run.accum_grad_iters 1 \
    run.num_workers 12 \
    run.seed 42 \
    run.evaluate True \
    run.valid_splits "['test']" \
    run.report_metric True \
    run.prefix test \
    run.resume_ckpt_path ${checkpoint_path}

