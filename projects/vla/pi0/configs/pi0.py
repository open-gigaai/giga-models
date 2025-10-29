data_paths = [
    './lerobot_data/',
]
data_or_config = []
for data_path in data_paths:
    data_or_config.append(
        dict(
            _class_name='LeRobotDataset',
            data_path=data_path,
            delta_info=dict(
                action=50,
            ),
            meta_name='meta',
        )
    )
config = dict(
    runners=['pi0.Pi0Trainer'],
    project_dir='./experiments/vla/pi0/base_debug',
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type='FSDP',
        fsdp_config=dict(
            fsdp_version='2',
            fsdp_auto_wrap_policy='TRANSFORMER_BASED_WRAP',
            fsdp_transformer_layer_cls_to_wrap='SiglipEncoderLayer,GemmaDecoderLayerWithExpert',
            fsdp_cpu_ram_efficient_loading='false',
            fsdp_state_dict_type='FULL_STATE_DICT',
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            batch_size_per_gpu=32,
            num_workers=16,
            transform=dict(
                type='Pi0Transform',
                adapt_to_pi=True,
                use_delta_joint_actions=True,
                norm_stats_path='./lerobot_data/meta/giga_pi05_norm_stats.json',
                image_cfg=dict(
                    resize_imgs_with_padding=[224, 224],
                    enable_image_aug=True,
                    present_img_keys=['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist'],
                ),
                prompt_cfg=dict(
                    tokenizer_model_path='google/paligemma-3b-pt-224',
                    max_length=48,
                ),
            ),
            sampler=dict(
                type='DefaultSampler',
                shuffle=True,
            ),
        ),
    ),
    models=dict(
        pretrained='./checkpoints/torch_pi0_base',
    ),
    optimizers=dict(
        type='AdamW',
        betas=(0.9, 0.95),
        lr=2.5e-5,
        eps=1e-8,
        weight_decay=1e-10,
    ),
    schedulers=dict(
        type='WarmupCosineScheduler',
        warmup_steps=1000,
        decay_steps=30000,
        end_value=0.1,  # decay_lr = optimizers.lr * end_value
    ),
    train=dict(
        resume=False,
        max_steps=50000,
        gradient_accumulation_steps=1,
        mixed_precision='no',  # Apply non-automatic mixed precision training.
        checkpoint_interval=1000,
        checkpoint_total_limit=5,
        checkpoint_keeps=[10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000],
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        dynamo_config=dict(backend='inductor'),
        activation_checkpointing=True,
        activation_class_names=[
            'SiglipEncoderLayer__##__16',
        ],
    ),
)
