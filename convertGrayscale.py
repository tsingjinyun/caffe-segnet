#!/usr/bin/env python
import os
import numpy as np
from itertools import izip
from argparse import ArgumentParser
from collections import OrderedDict
from skimage.io import ImageCollection, imsave
from skimage.transform import resize


camvid_colors = OrderedDict([
    ("Animal", np.array([64, 128, 64], dtype=np.uint8)),
    ("Archway", np.array([192, 0, 128], dtype=np.uint8)),
    ("Bicyclist", np.array([0, 128, 192], dtype=np.uint8)),
    ("Bridge", np.array([0, 128, 64], dtype=np.uint8)),
    ("Building", np.array([128, 0, 0], dtype=np.uint8)),
    ("Car", np.array([64, 0, 128], dtype=np.uint8)),
    ("CartLuggagePram", np.array([64, 0, 192], dtype=np.uint8)),
    ("Child", np.array([192, 128, 64], dtype=np.uint8)),
    ("Column_Pole", np.array([192, 192, 128], dtype=np.uint8)),
    ("Fence", np.array([64, 64, 128], dtype=np.uint8)),
    ("LaneMkgsDriv", np.array([128, 0, 192], dtype=np.uint8)),
    ("LaneMkgsNonDriv", np.array([192, 0, 64], dtype=np.uint8)),
    ("Misc_Text", np.array([128, 128, 64], dtype=np.uint8)),
    ("MotorcycleScooter", np.array([192, 0, 192], dtype=np.uint8)),
    ("OtherMoving", np.array([128, 64, 64], dtype=np.uint8)),
    ("ParkingBlock", np.array([64, 192, 128], dtype=np.uint8)),
    ("Pedestrian", np.array([64, 64, 0], dtype=np.uint8)),
    ("Road", np.array([128, 64, 128], dtype=np.uint8)),
    ("RoadShoulder", np.array([128, 128, 192], dtype=np.uint8)),
    ("Sidewalk", np.array([0, 0, 192], dtype=np.uint8)),
    ("SignSymbol", np.array([192, 128, 128], dtype=np.uint8)),
    ("Sky", np.array([128, 128, 128], dtype=np.uint8)),
    ("SUVPickupTruck", np.array([64, 128, 192], dtype=np.uint8)),
    ("TrafficCone", np.array([0, 0, 64], dtype=np.uint8)),
    ("TrafficLight", np.array([0, 64, 64], dtype=np.uint8)),
    ("Train", np.array([192, 64, 128], dtype=np.uint8)),
    ("Tree", np.array([128, 128, 0], dtype=np.uint8)),
    ("Truck_Bus", np.array([192, 128, 192], dtype=np.uint8)),
    ("Tunnel", np.array([64, 0, 64], dtype=np.uint8)),
    ("VegetationMisc", np.array([192, 192, 0], dtype=np.uint8)),
    ("Wall", np.array([64, 192, 0], dtype=np.uint8)),
    ("Void", np.array([0, 0, 0], dtype=np.uint8))
])


def convert_label_to_grayscale(im):
    out = (np.ones(im.shape[:2]) * 255).astype(np.uint8)
    for gray_val, (label, rgb) in enumerate(camvid_colors.items()):
        match_pxls = np.where((im == np.asarray(rgb)).sum(-1) == 3)
        out[match_pxls] = gray_val
    assert (out != 255).all(), "rounding errors or missing classes in camvid_colors"
    return out.astype(np.uint8)


def make_parser():
    parser = ArgumentParser()
    parser.add_argument(
        'label_dir',
        help="Directory containing all RGB camvid label images as PNGs"
    )
    parser.add_argument(
        'out_dir',
        help="""Directory to save grayscale label images.
        Output images have same basename as inputs so be careful not to
        overwrite original RGB labels""")
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    labs = ImageCollection(os.path.join(args.label_dir, "*"))
    os.makedirs(args.out_dir)
    for i, (inpath, im) in enumerate(izip(labs.files, labs)):
        print i + 1, "of", len(labs)
        # resize to caffe-segnet input size and preserve label values
        resized_im = (resize(im, (360, 480), order=0) * 255).astype(np.uint8)
        out = convert_label_to_grayscale(resized_im)
        outpath = os.path.join(args.out_dir, os.path.basename(inpath))
        imsave(outpath, out)
