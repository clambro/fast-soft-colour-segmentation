import os.path
import warnings
from argparse import ArgumentParser

import numpy as np
from skimage import io
from skimage.transform import rescale

from model import FSCSModel


def main(img_path, out_path, scale):
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f'The file {img_path} was not found.')
    if not os.path.isdir(out_path):
        raise NotADirectoryError(f'The directory {out_path} is either not valid or does not exist.')
    if not 0 < scale <= 1:
        raise ValueError('The scale parameter must be in (0, 1].')

    img = io.imread(img_path)
    img = rescale(img, scale, multichannel=True)  # Rescaling with scale=1 converts values to floats between 0 and 1.

    f = FSCSModel()
    channels = f.optimize_for_one_image(img)

    alpha = np.asarray([ch[..., 3] for ch in channels])
    colour = np.asarray([ch[..., :3] for ch in channels])
    recon_img = np.sum(alpha[..., None] * colour, axis=0)

    img_name = '.'.join('.'.split(os.path.basename(img_path)[:-1]))
    recon_path = os.path.join(out_path, f'reconstructed_{img_name}.png')
    io.imsave(recon_path, np.rint(255 * recon_img).astype(np.uint8))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')  # Suppress low contrast image warnings.
        for i, ch in enumerate(channels):
            ch_path = os.path.join(out_path, f'channel_{i}_{img_name}.png')
            io.imsave(ch_path, np.rint(255 * ch).astype(np.uint8))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('img_path', help='The path to the image to optimize on.')
    parser.add_argument('out_path', help='The path to which the results will be saved.')
    parser.add_argument('scale', type=float, default=1,
                        help='Rescaling factor from 0 to 1 to downsample the image if necessary. Default is 1.')
    args = parser.parse_args()

    main(args.img_path, args.out_path, args.scale)
