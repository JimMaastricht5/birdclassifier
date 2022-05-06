#!/usr/bin/env python3
import multiprocessing
import pandas as pd


class AssetStream:
    # item format is expected to be a tuple with
    # event_num: unique int key for event
    # type: end_process, motion, prediction, prediction_final, flush
    #   end_process or None: stops the process and free the creator's block on join()
    #   motion, prediction, prediction_final: all contain text for the web site to display
    #   flush: writes the current set of in memory content to a file for web site display
    # Date time: string
    # Prediction: string
    # Image_name: string, image name on disk

    def __init__(self, queue):
        self.queue = queue
        self.df = pd.DataFrame({
                           'event_num': pd.Series(dtype='int'),
                           'type': pd.Series(dtype='str'),
                           'Date Time': pd.Series(dtype='str'),
                           'Prediction': pd.Series(dtype='float'),
                           'Image_Name': pd.Series(dtype='str')})

    def request_handler(self):
        while True:
            item = self.queue.get()  # get the next item in the queue to write to disk
            event_type = item[1]
            if event_type == 'end_process' or event_type is None:
                break
            elif event_type == 'flush':
                self.df.to_csv('/home/pi/birdclass/webstream.csv')
            else:
                self.df.append(item)
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
