#!/usr/bin/env python3
import multiprocessing
import pandas as pd


class WebStream:
    # item format is expected to be a tuple with
    # event_num: unique int key for event
    # type: end_process, motion, prediction, prediction_final, flush, message
    #   end_process or None: stops the process and free the creator's block on join()
    #   motion, prediction, prediction_final, message: all contain text for the web site to display
    #   flush: writes the current set of in memory content to a file for web site display
    # Date time: string
    # Prediction: string
    # Image_name: string, image name on disk
    def __init__(self, queue):
        print('stream init')
        self.queue = queue
        self.df = pd.DataFrame({
                           'event_num': pd.Series(dtype='int'),
                           'type': pd.Series(dtype='str'),
                           'Date Time': pd.Series(dtype='str'),
                           'Message': pd.Series(dtype='float'),
                           'Image_Name': pd.Series(dtype='str')})
        print('end stream init')

    def request_handler(self):
        while True:
            item = self.queue.get()  # get the next item in the queue to write to disk
            if item is None:  # poison pill, end the process
                break
            elif item[1] == 'flush':  # event type is flush
                self.df.to_csv('/home/pi/birdclass/webstream.csv')
            else:  # any other event type
                self.df.append(item)
        return


class WebStreamController:
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.web_stream = WebStream(queue=self.queue)
        print('setting up multiprocessing')
        self.p_write_asset = multiprocessing.Process(target=self.web_stream.request_handler(), args=(), daemon=True)
        print('end init controller')

    def start_stream(self):
        self.p_write_asset.start()
        return

    def end_stream(self):
        try:
            self.queue.put(None)
            if self.p_write_asset.is_alive():
                print('waiting for web stream to finish processing queue....')
                self.p_write_asset.join()
        finally:
            pass
            # p.terminate()
        return


def main():
    web_stream_controller = WebStreamController()
    print('starting')
    web_stream_controller.start_stream()
    print('ending')
    web_stream_controller.end_stream()


if __name__ == '__main__':
    main()
