""" Usage:
   pr_plot --in=DIR_NAME --out=OUTPUT_FILENAME 

Options:
  --in=DIR_NAME            Folder in which to search for *.dat files, all of which should be in a P/R column format (outputs from benchmark.py)
  --out=OUTPUT_FILENAME    Output filename, filetype will determine the format. Possible formats: pdf, pgf, png


"""

import os
import ntpath
import numpy as np
from glob import glob
from docopt import docopt
import matplotlib.pyplot as plt
import logging
import ipdb
logging.basicConfig(level = logging.INFO)

plt.rcParams.update({'font.size': 14})

def trend_name(path):
    ''' return a system trend name from dat file path '''
    head, tail = ntpath.split(path)
    ret = tail or ntpath.basename(head)
    return ret.split('.')[0]

def get_pr(path):
    ''' get PR curve from file '''
    with open(path) as fin:
        # remove header line
        fin.readline()
        prc = list(zip(*[[float(x) for x in line.strip().split('\t')] for line in fin]))
        p = prc[0]
        r = prc[1]
        return p, r
    
if __name__ == '__main__':
    args = docopt(__doc__)
    input_folder = args['--in']
    output_file = args['--out']
    
    # plot graphs for all *.dat files in input path
    files = glob(os.path.join(input_folder, '*.dat'))
    for _file in files:
        p, r = get_pr(_file)
        name = trend_name(_file)
        plt.plot(r, p, label = name)

    # Set figure properties and save
    logging.info("Plotting P/R graph to {}".format(output_file))
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 0.6])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig(output_file)
