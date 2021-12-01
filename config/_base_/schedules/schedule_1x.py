# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.002,betas=(0.9, 0.99), eps=1e-08, weight_decay=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     min_lr_ratio=1e-2,
#     by_epoch=True)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7,11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
