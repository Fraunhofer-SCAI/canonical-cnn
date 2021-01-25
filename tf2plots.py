#!/usr/bin/env python3

import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tikzplotlib import save as tikz_save

def convert_Log2Df(events, tag_req):
    size_guidance = {
        'COMPRESSED_HISTOGRAMS': 0,
        'IMAGES': 0,
        'AUDIO': 0,
        'SCALARS': 0,
        'HISTOGRAMS': 0,
    }
    vals = []
    step = []
    try:
        event_acc = EventAccumulator(events, size_guidance)
        event_acc.Reload()
        avail_tags = event_acc.Tags()['scalars']
        for each_tag in avail_tags:
            if each_tag == tag_req:
                event_list = event_acc.Scalars(each_tag)
                vals = list(map(lambda x: x.value, event_list))
                step = list(map(lambda x:x.step, event_list))
        plt.figure()
        plt.plot(step, vals, label=tag_req)
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel(tag_req)
        plt.legend()
        tikz_save('plot.tex')
    except Exception:
        print('Exception occured')



def main(path, tag_req):
    if os.path.isfile(path):
        events = path
    else:
        print('No file found...')

    if events is not None:
        print("Reading event file...")
        log_df = convert_Log2Df(events, tag_req)

if __name__ == "__main__":
    path = sys.argv[1]
    tag_req = sys.argv[2]
    main(path, tag_req)