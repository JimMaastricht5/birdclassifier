#!/usr/bin/env python3
import multiprocessing
import datetime
import pandas as pd
import numpy as np


# item format is expected to be a tuple with
# [0]: command: end_process, motion, prediction, prediction_final, motion_jpg, prediction_gif, flush
# end_process: stops the process and free the creator's block on join()
# motion, prediction, prediction_final: all contain text for the web site to display
# motion_jpg, prediction_gif: contain image data
# flush: writes the current set of in memory content to a file for web site display
# [1]: data (text or image)
class AssetStream:
    def __init__(self, queue):
        self.queue = queue
        self.df = pd.DataFrame({'type': pd.Series(dtype='str'),
                           'Date Time': pd.Series(dtype='str'),
                           'Prediction': pd.Series(dtype='float'),
                           'Image_Name': pd.Series(dtype='str'),
                           'Image_Type': pd.Series(dtype='str')})

    def request_handler(self):
        while True:
            item = self.queue.get()  # get the next item in teh queue to write to disk
            if item[0] == 'end_process':
                break
            elif item[0] == 'motion':
                self.df.append(item[0], item[1], 0, '', '')
                pass
            elif item[0] == 'prediction' or 'prediction_final':
                pass
            elif item[0] == 'motion_jpg' or item[0] == 'prediction_gif':
                pass
            elif item[0] == 'flush':
                pass
        return


def main():
    queue = multiprocessing.Queue()
    asset_stream = AssetStream(queue=queue)
    p_write_asset = multiprocessing.Process(target=asset_stream.request_handler(), args=(), daemon=True)
    p_write_asset.start()
    p_write_asset.join()
    # p.terminate(), p.is_alive()


if __name__ == '__main__':
    main()
