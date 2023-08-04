import os
from ulw_data.preprocessing.preprocess_pipeline import preprocess
from ulw_data.preprocessing.WMH_Chal_file_parser import get_files_dict
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='preprocess WMH challenge data')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--output_folder_root', type=str)
    parser.add_argument('--out_spacing', default="1.,1.,3.", type=str, help="output spacing used in the resampling process. Provide as a string of comma separated floats, no spaces, i.e '1.,1.,3.")
    parser.add_argument('--lower_norm_percentile', default=0, type=float)
    parser.add_argument('--upper_norm_percentile', default=100, type=float)
    
    return parser


def main(args):
    ds_path = args.dataset_path
    out_spacing = [float(x) for x in args.out_spacing.split(",")]
    print("using outspacing: ", out_spacing)
    output_folder_root = args.output_folder_root

    output_folder = os.path.sep.join([output_folder_root, 'preprocessed', 'individual_files'])
    try:
        os.makedirs(output_folder)
    except FileExistsError:
        print("output folder already exists, continuing")

    files_dict = get_files_dict(ds_path, output_folder)
    
    for fid, f_dict in files_dict.items():
        print(f'#  processing {fid}')
        preprocess(f_dict, out_spacing=out_spacing, lower_norm_percentile=args.lower_norm_percentile, upper_norm_percentile=args.upper_norm_percentile)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)