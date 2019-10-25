"""
Example:
python evaluations/gen_images_at_iters.py --n_zs=10 --snapshot=ResNetGenerator_850000.npz --classes 1 --iters 500 20500 2000
"""

import os, sys, time
import shutil
import numpy as np
import argparse
import chainer
from PIL import Image

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from evaluation import gen_images_with_condition
import yaml
import source.yaml_utils as yaml_utils
from source.miscs.random_samples import sample_continuous


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='./results/gans')
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--n_intp', type=int, default=5)
    parser.add_argument('--n_zs', type=int, default=5)
    parser.add_argument('--classes', type=int, nargs="*", default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--iters', type=int, nargs="*", default=None)
    args = parser.parse_args()
    chainer.cuda.get_device(args.gpu).use()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    gen = load_models(config)
    gen.to_gpu()
    out = args.results_dir
    chainer.serializers.load_npz(args.snapshot, gen)
    np.random.seed(args.seed)
    xp = gen.xp
    n_images = args.n_zs * args.n_intp
    imgs = []
    classes = tuple(args.classes) if args.classes is not None else [np.random.randint(1000), np.random.randint(1000)]
    iters = tuple(args.iters)
    snapshots = []
    snapshot_base = '_'.join(args.snapshot.split('_')[:-1])
    for i in range(iters[0], iters[1], iters[2]):
        snapshots.append('{}_{}.npz'.format(snapshot_base, str(i)))
        
    for snapshot in snapshots:
        chainer.serializers.load_npz(snapshot, gen)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen_images_with_condition(gen, c=classes[0], n=args.n_zs, batchsize=args.n_zs)
        imgs.append(x)
    img = np.stack(imgs)
    _, _, _, h, w = img.shape
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape((len(snapshots) * h, args.n_zs * w, 3))

    save_path = os.path.join(out, 'snapshot_{}-{}-{}.png'.format(iters[0], iters[1], iters[2]))
    if not os.path.exists(out):
        os.makedirs(out)
    Image.fromarray(img).save(save_path)


if __name__ == '__main__':
    main()