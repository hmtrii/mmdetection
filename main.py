from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv import Config

# Get configuration
cfg = Config.fromfile('/mnt/c/Users/Dat/Desktop/ML/medical-project/repo/mmdetection/baseline_config.py')
print(cfg.pretty_text)
# Build dataset
datasets = [build_dataset(cfg.data.train)]
print(datasets)
# aaa
# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES
train_detector(model, datasets, cfg, distributed=False, validate=True)
