#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2020-11-19
Purpose: Thermal mean temperature extraction
"""

import argparse
import os
import sys
from osgeo import gdal
import numpy as np
import pandas as pd
from scipy import stats
import skimage.color
import skimage.filters
import skimage.io
import multiprocessing
import warnings
import tifffile as tifi
warnings.filterwarnings("ignore")


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Mean temperature extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('tif',
                        nargs='+',
                        metavar='tif',
                        help='TIF image to analyze for plot temp')

    parser.add_argument('-o',
                        '--outdir',
                        help='Output directory',
                        metavar='outdir',
                        type=str,
                        default='peakstemp_out')

    parser.add_argument('-d',
                        '--date',
                        help='Date of UAV flight',
                        metavar='date',
                        type=str,
                        required=True)

    return parser.parse_args()


# --------------------------------------------------
def process_image(img):
    args = get_args()
    temps_dict = {}
    cont_cnt = 0

    plot = img.split('/')[-2]
    sigma = float(1)
    df = pd.DataFrame()

    try:
        cont_cnt += 1
        #image = skimage.io.imread(fname=img)
        a_img = tifi.imread(img)
        a_img = np.array(a_img)
        image = a_img[~np.isnan(a_img)]
        blur = skimage.color.rgb2gray(image)
        blur = skimage.filters.gaussian(image, sigma=sigma)
        t = skimage.filters.threshold_otsu(blur)

        mask = blur < t
        sel = np.zeros_like(image)
        sel[mask] = image[mask]
        sel[sel == 0] = np.nan

        mask2 = blur > t
        sel2 = np.zeros_like(image)
        sel2[mask2] = image[mask2]
        sel2[sel2 == 0] = np.nan

        soil_temp = np.nanmean(sel2)
        plant_temp = np.nanmean(sel)
        img_temp = np.nanmean(image)
        print(soil_temp)

        print(f'Soil temp: {soil_temp}\nPlant temp: {plant_temp}\n')

        temps_dict[cont_cnt] = {'date': str(args.date),
                    'plot': str(plot),
                    'soil_mean_temp': float(soil_temp),
                    'plant_mean_temp': float(plant_temp),
                    'image_mean_temp': float(img_temp)}

        df = pd.DataFrame.from_dict(temps_dict, orient='index', columns=['date',
                                                                        'plot',
                                                                        'soil_mean_temp',
                                                                        'plant_mean_temp',
                                                                        'image_mean_temp']).set_index('date')

    except:
        pass

    return df


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    major_df = pd.DataFrame()

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        df = p.map(process_image, args.tif)
        major_df = major_df.append(df)

    out_path = os.path.join(args.outdir, f'{args.date}_meantemp.csv')
    major_df.to_csv(out_path)

    print(f'Done. Check output in {args.outdir}')


# --------------------------------------------------
if __name__ == '__main__':
    main()
