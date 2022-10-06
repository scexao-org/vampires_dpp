#!/usr/bin/env python

from argparse import ArgumentParser
from astropy.io import fits
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path


def calib(args):
    if args.dark:
        master_dark = fits.getdata(args.dark)
    if args.flat:
        master_flat = fits.getdata(args.flat)
    for filename in tqdm.tqdm(args.filename):
        _path = Path(filename)
        cube, header = fits.getdata(_path, header=True)
        out = np.empty_like(cube, "f4")
        if args.dark:
            out = cube - master_dark
        if args.flat:
            out /= master_flat

        if args.out:
            outname = args.out
        else:
            outname = _path.with_name(f"{_path.stem}_calib{_path.suffix}")
        fits.writeto(outname, out, header=header, overwrite=True)


def make_dark(args):
    for filename in tqdm.tqdm(args.filename):
        _path = Path(filename)
        cube, header = fits.getdata(_path, header=True)
        master_dark = np.median(cube, axis=0)
        if args.out:
            outname = args.out
        else:
            outname = _path.with_name(f"{_path.stem}_master_dark{_path.suffix}")
        fits.writeto(outname, master_dark, header=header, overwrite=True)


def make_flat(args):
    if args.dark:
        master_dark = fits.getdata(args.dark)
    for filename in tqdm.tqdm(args.filename):
        _path = Path(filename)
        cube, header = fits.getdata(_path, header=True)
        if args.dark:
            cube -= master_dark
        master_flat = np.median(cube, axis=0)
        master_flat /= np.median(master_flat)
        if args.out:
            outname = args.out
        else:
            outname = _path.with_name(f"{_path.stem}_master_flat{_path.suffix}")
        fits.writeto(outname, master_flat, header=header, overwrite=True)


def combine(args):
    cubes = [fits.getdata(fname) for fname in args.filename]
    reduced = np.median(np.asarray(cubes), axis=0)
    fits.writeto(args.out, reduced, overwrite=True)


parser = ArgumentParser()
subparsers = parser.add_subparsers()

parser_calib = subparsers.add_parser("calib")
parser_calib.add_argument("filename", nargs="+", help="FITS files to calibrate")
parser_calib.add_argument("-d", "--dark", help="dark frame to subtract from raw frames")
parser_calib.add_argument(
    "-f", "--flat", help="normalized flat frame for correcting frames"
)
parser_calib.add_argument("--discard", default=0, help="discard initial frames")
parser_calib.add_argument(
    "-o", "--out", help="name of output file, by default will append name with `_calib`"
)
parser.set_defaults(func=calib)

# making darks
parser_darks = subparsers.add_parser("make_dark")
parser_darks.add_argument(
    "filename", nargs="+", help="FITS files to create dark frames from"
)
parser_darks.add_argument(
    "-o",
    "--out",
    help="name of output file, by default will append name with `_master_dark`",
)
parser_darks.set_defaults(func=make_dark)

# making flats
parser_flat = subparsers.add_parser("make_flat")
parser_flat.add_argument(
    "filename", nargs="+", help="FITS files to create dark frames from"
)
parser_flat.add_argument("-d", "--dark", help="dark frame to subtract from raw frames")
parser_flat.add_argument(
    "-o",
    "--out",
    help="name of output file, by default will append name with `_master_flat`",
)
parser_flat.set_defaults(func=make_flat)

# combining frames
parser_comb = subparsers.add_parser("combine")
parser_comb.add_argument("filename", nargs="+", help="FITS files to median combine")
parser_comb.add_argument("-o", "--out", required=True)
parser_comb.set_defaults(func=combine)


def main():
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
