CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    --master_port=34658 \
    train.py \
    --cfg-path lavis/projects/malmm/qa_msvd.yaml \
    --options \
    model.arch blip2_vicuna_instruct_apt \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 10 \
    model.num_frames 20 \
    run.init_lr 1e-4 \
    run.max_epoch 5 \
    run.num_beams 5 \
    run.batch_size_train 32 \
    run.batch_size_eval 32 \
    run.accum_grad_iters 1 \
    run.num_workers 12 \
    run.seed 42 \
    run.evaluate True \
    run.valid_splits "['val', 'test']" \
    run.report_metric True \
    run.prefix test \
    run.resume_ckpt_path '/lrh/MA-LMM-shl/lavis/output/msvd_qa/blip2_vicuna_instruct_apt_vicuna7b/train/b32_e10_lr0.0001_wd0.05_q32_f20_fb10_freezevit/checkpoint_best.pth'

