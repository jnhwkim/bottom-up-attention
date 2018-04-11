import time, os, sys
import argparse
import numpy as np
import csv
import scipy.io

csv.field_size_limit(sys.maxsize)

N_SPLIT_DATA = {'train': 29783, 
                'val': 1000, 
                'test': 1000 }

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Split data (train, val, test) from whole flickr30k data')
    parser.add_argument('--file', dest='file',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--split', dest='split',
                        help='split data file path (mat file)',
                        default=None, type=str)

    args = parser.parse_args()
    return args

def load_split_data(file_path):
    mat = scipy.io.loadmat(file_path)
    split_info = {
        'train': mat['trainfns'],
        'val': mat['valfns'],
        'test': mat['testfns'],
    }
    return split_info

def split_tsvs(file_path, split_info=None):

    prefixes = N_SPLIT_DATA.keys()
    out_tsv = None
    split_idx = 0
    count = 0
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)

    if split_info is None:
        with open(file_path) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)

            for item in reader:
                if out_tsv == None:
                    prefix = prefixes[split_idx]
                    num_data = N_SPLIT_DATA[prefix]
                    out_file = prefix + '_' + file_name
                    out_path = os.path.join(file_dir, out_file)
                    out_tsv = open(out_path, 'wb')
                    writer = csv.DictWriter(out_tsv, delimiter = '\t', fieldnames = FIELDNAMES)

                try:
                    writer.writerow(item)
                except Exception as e:
                    print(e)
                count += 1

                if count == num_data:
                    count = 0
                    split_idx += 1
                    out_tsv.close()
                    out_tsv = None

        if out_tsv is not None and not out_tsv.closed:
            out_tsv.close()
    else:
        output_ios = {}
        output_tsv = {}
        for mode in split_info.keys():
            output_name = mode + '_' + file_name
            output_path = os.path.join(file_dir, output_name)
            output_ios[mode] = open(output_path, 'wb')
            output_tsv[mode] = csv.DictWriter(output_ios[mode], delimiter = '\t', fieldnames = FIELDNAMES)

        test_list = split_info['test']
        val_list = split_info['val']
        with open(file_path) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)

            for item in reader:
                image_id = item['image_id']
                if image_id in test_list:
                    output_tsv['test'].writerow(item)
                elif image_id in val_list:
                    output_tsv['val'].writerow(item)
                else:
                    output_tsv['train'].writerow(item)

        for io in output_ios.values():
            io.close()


     
if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    split_info = None
    if args.split is not None:
        split_info = load_split_data(args.split)

    split_tsvs(args.file, split_info)