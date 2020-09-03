#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2020-09-03
Purpose: Extract plot-level temperature values for UAV imagery
"""

import argparse
import os
import sys
from osgeo import gdal
import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='UAV thermal extraction',
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
def main():
    """Make a jazz noise here"""

    args = get_args()
    temps_dict = {}
    cnt = 0

    for img in args.tif:
        cnt += 1
        g_img = gdal.Open(img)
        a_img = g_img.GetRasterBand(1).ReadAsArray()
        plot_temp = np.nanmean(a_img)

        mean_list = []

        for i in a_img:
            mean = i.mean()
            mean_list.append(mean)

        avg = sum(mean_list)/len(mean_list)
        avg = avg + 1

        a_img[a_img > avg] = np.nan

        plant_temp = np.nanmean(a_img)
        diff = abs(float(plot_temp) - float(plant_temp))

        temps_dict[cnt] = {
                'filename': os.path.basename(img),
                'plot_temp': plot_temp,
                'plant_temp': plant_temp,
                'plot_plant_diff': diff
        }

        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)

        df = pd.DataFrame.from_dict(temps_dict, orient='index')
        df.to_csv(os.path.join(args.outdir, args.date + '_peakstemp.csv'))

        print(f'Plot temp: {plot_temp}\nPlant temp: {plant_temp}\nDifference: {diff}\n')

    print(f'Done. Check output in {args.outdir}')


# --------------------------------------------------
if __name__ == '__main__':
    main()