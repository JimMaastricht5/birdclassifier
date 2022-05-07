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


class Controller:
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.web_stream = WebStream(queue=self.queue)
        self.p_web_stream = multiprocessing.Process(target=self.web_stream.request_handler, args=(), daemon=True)
        self.df = pd.DataFrame({
                           'event_num': pd.Series(dtype='int'),
                           'type': pd.Series(dtype='str'),
                           'Date Time': pd.Series(dtype='str'),
                           'Message': pd.Series(dtype='str'),
                           'Image_Name': pd.Series(dtype='str')})

    def start_stream(self):
        self.p_web_stream.start()
        return

    def message(self, message):
        print(message)
        item = [0, 'message', datetime.datetime.now().strftime("%H:%M:%S"), message, '']
        self.queue.put(item)
        return

    def flush(self):
        item = [0, 'flush', datetime.datetime.now().strftime("%H:%M:%S"), '', '']
        self.queue.put(item)

    def end_stream(self):
        try:
            self.queue.put(None)
            if self.p_web_stream.is_alive():
                print('waiting for web stream to finish processing queue....')
                self.p_web_stream.join()
        finally:
            print('')
        return


def main():
    web_stream = Controller()
    web_stream.start_stream()

    web_stream.end_stream()


if __name__ == '__main__':
    main()


# simple sample queue code
# from multiprocessing import Process, Queue
#
# class WebWriter:
#     def __init__(self, queue):
#         self.queue = queue
#
#     def request_handler(self):
#         while True:
#             message = self.queue.get()
#             if message is None:
#                 break
#             print(message)
#
# if __name__ == '__main__':
#     # Create multiprocessing queue
#     queue = Queue()
#     web_writer = WebWriter(queue=queue)
#     p = Process(target=web_writer.request_handler, args=())
#     p.start()
#     for i in range(10):
#         queue.put(f'{i}ith message')
#     queue.put(None)
