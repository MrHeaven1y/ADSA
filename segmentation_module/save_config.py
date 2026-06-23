from .segmentation.datasets import benchmark_num_workers
from .segmentation.utils import save_config, Utils

if __name__ == '__main__':

    config_path = ".\artifacts\default_configs.json"
    utils = Utils(CONFIG_PATH=config_path)
    config = utils.CONFIG

    config.setdefault('images_dir',      '/kaggle/input/datasets/huanghanchina/pascal-voc-2012/VOC2012/JPEGImages')
    config.setdefault('mask_dir',        '/kaggle/input/datasets/dibyenducontroversy/voc-masks')
    config.setdefault('checkpoint_dir',  '/kaggle/working/checkpoints')  # periodic + resume saves
    config.setdefault('best_model_dir',  '/kaggle/working/best_model')   # ← directly downloadable
    config.setdefault('max_samples',     5000)

    config.setdefault('img_size',        128)     # ← 128×128
    config.setdefault('latent_channels', 128)     # halved from 256 — right for 128px bottleneck
    config.setdefault('reduction_rate',  8)       # SE ratio (8 works better at smaller maps)

    config.setdefault('batch_size',      32)      # 128px fits more per GPU than 256px
    config.setdefault('max_iterations',  50)
    config.setdefault('lr',              1e-3)
    config.setdefault('weight_decay',    1e-4)
    config.setdefault('grad_clip',       1.0)
    config.setdefault('accum_steps',     1)       # no accum needed at batch_size=32
    config.setdefault('split_size',      0.85)
    config.setdefault('patience',        10)
    config.setdefault('min_delta',       1e-4)
    config.setdefault('resume',          None)
    config.setdefault('save_every',      5)

    config.setdefault('lambda_l1',          1.0)
    config.setdefault('lambda_ssim',        0.5)  # ↑ SSIM matters more at lower res
    config.setdefault('lambda_perceptual',  0.1)
    config.setdefault('lambda_style',       0.0)  # disabled
    config.setdefault('lambda_edge',        0.3)  # preserve obj/bg boundary for watermarking

    config['num_workers'] = benchmark_num_workers(
        config['batch_size'],
        config['img_size']
    )

    save_config(config, utils.CONFIG_PATH)