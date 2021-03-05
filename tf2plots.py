#!/usr/bin/env python3
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
#from tikzplotlib import save as tikz_save

def convert_log2df(events, tag_req):
    """This method reads the tensorboard event file and extract required
        tag

    Args:
        events (str): Path for tensorboard event file
        tag_req (str): Required tag to extract from event file

    Returns:
        list: List of extracted tag values
    """
    
    size_guidance = {
        'COMPRESSED_HISTOGRAMS': 0,
        'IMAGES': 0,
        'AUDIO': 0,
        'SCALARS': 0,
        'HISTOGRAMS': 0,
    }
    vals = []
    step = []
    # event accumulator to read the file
    try:
        event_acc = EventAccumulator(events, size_guidance)
        event_acc.Reload()
        avail_tags = event_acc.Tags()['scalars']
        for each_tag in avail_tags:
            # Extract all values in the tag
            if each_tag == tag_req:
                event_list = event_acc.Scalars(each_tag)
                vals = list(map(lambda x: x.value, event_list))
                step = list(map(lambda x:x.step, event_list))

    except Exception:
        print('Exception occured')

    return vals



def main(path, tag_req):
    """This method extract all events file in subdirectories and saves
        the mean and standard deviation of all event file to csv file
    Args:
        path (str): Path for all subdirectories of same experiment
        tag_req ([type]): Required tag to extract from event file
    """
    
    file_list = []
    # Extract all the files in all the subdirectories
    for (dir_, _, fname) in os.walk(path):
        file_list += [os.path.join(dir_, file_) for file_ in fname]

    if file_list is not None:
        total_values = []
        # Extract all the values in every event file
        for index, each_file in enumerate(file_list):
            print("Reading event file-",index+1,"...")
            values = convert_log2df(each_file, tag_req)
            if len(values) != 0:
                total_values.append(values)
        total_values = np.asarray(total_values)
        # Convert np array to pd dataframe 
        mean = np.expand_dims(np.mean(total_values, axis=0), axis=1)
        std = np.expand_dims(np.std(total_values, axis=0), axis=1)
        total = np.hstack((mean, std))
        dframe = pd.DataFrame(data= total, columns=["mean", "std"])
        # Save the pd file as CSV
        fname = path+'_mean_std.csv'
        dframe.to_csv(fname)



if __name__ == "__main__":
    path = sys.argv[1]
    tag_req = sys.argv[2]
    main(path, tag_req)