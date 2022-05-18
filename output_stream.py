#!/usr/bin/env python3
import multiprocessing
import pandas as pd
import datetime
import os


class WebStream:
    # item format is expected to be a tuple with
    # event_num: unique int key for event
    # types: end_process, motion, prediction, prediction_final, flush, message
    #   end_process or None: stops the process and free the creator's block on join()
    #   motion, prediction, prediction_final, message: all contain text for the website to display
    #   flush: writes the current set of in memory content to a file for website display
    # Date time: string
    # Prediction: string
    # Image_name: string, image name on disk
    def __init__(self, queue, path=os.getcwd()):
        self.queue = queue
        self.path = path
        print(self.path)
        self.asset_path = self.path + '/assets'
        print(self.asset_path)
        self.df_list = []
        self.df = pd.DataFrame({
                           'Event Num': pd.Series(dtype='int'),
                           'type': pd.Series(dtype='str'),
                           'Date Time': pd.Series(dtype='str'),
                           'Message': pd.Series(dtype='str'),
                           'Image Name': pd.Series(dtype='str')})
        self.df_occurrences = pd.DataFrame({
                           'Species': pd.Series(dtype='str'),
                           'Date Time': pd.Series(dtype='str')})

    def request_handler(self):
        try:
            while True:
                item = self.queue.get()  # get the next item in the queue to write to disk
                if item is None:  # poison pill, end the process
                    break  # end process

                # print('Message Type:',item[1])  # print message to terminal
                if item[1] == 'flush':  # event type is flush
                    self.df = pd.DataFrame(self.df_list,
                                           columns=['Event Num', 'type', 'Date Time', 'Message', 'Image Name'])
                    self.df.to_csv(f'{self.path}/webstream.csv')
                elif item[1] == 'occurrences':
                    print('writing occurrences to web')
                    print('item[3]:', item[3])  # show list of species occurrences
                    self.df_occurrences = pd.DataFrame(item[3], columns=['Species', 'Date Time'])
                    self.df_occurrences.to_csv(f'{self.path}/web_occurrences.csv')  # species, date time
                else:  # basic message or any other event type
                    print(item[3])  # print message
                    self.df_list.append(item)
        except Exception as e:
            print(e)
        return


class Controller:
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.web_stream = WebStream(queue=self.queue)
        self.p_web_stream = multiprocessing.Process(target=self.web_stream.request_handler, args=(), daemon=True)
        self.last_event_num = 0
        self.df = pd.DataFrame({
                           'Event Num': pd.Series(dtype='int'),
                           'type': pd.Series(dtype='str'),
                           'Date Time': pd.Series(dtype='str'),
                           'Message': pd.Series(dtype='str'),
                           'Image Name': pd.Series(dtype='str')})

    def start_stream(self):
        self.p_web_stream.start()
        return

    def message(self, message, event_num=0, msg_type='message', image_name='', flush=False):
        # print('web controller sending: ', message)
        event_num = self.last_event_num if event_num == 0 else event_num
        item = [event_num, msg_type, datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), message, image_name]
        self.queue.put(item)
        if flush:
            self.flush()
        self.last_event_num = event_num
        return

    def occurrences(self, occurrence_list):
        # print(occurrence_list)
        item = [0, 'occurrences', datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), occurrence_list]
        self.queue.put(item)
        return

    def flush(self):
        item = [0, 'flush', datetime.datetime.now().strftime("%H:%M:%S"), '', '']
        self.queue.put(item)

    def end_stream(self):
        self.flush()  # write any pending contents to disk
        try:
            self.queue.put(None)  # transmit poison pill to stop child process
            if self.p_web_stream.is_alive():  # wait for the child process to finish if it is still alive
                print('waiting for web stream to finish processing queue....')
                self.p_web_stream.join()
        finally:
            print('')
        return


def main():
    web_stream = Controller()
    web_stream.start_stream()
    web_stream.message('up and running')  # place message on queue for child process
    web_stream.message(event_num=1, msg_type='prediction', message='big fat robin 97.0%',
                       image_name='/home/pi/birdclass/first_img.jpg')
    web_stream.message(event_num=1, msg_type='prediction', message='big fat robin 37.0%',
                       image_name='/home/pi/birdclass/first_img.jpg')
    web_stream.message(event_num=1, msg_type='prediction', message='big fat robin 96.0%',
                       image_name='/home/pi/birdclass/first_img.jpg')
    web_stream.message(event_num=1, msg_type='final_prediction', message='big fat robin 74.0%',
                       image_name='/home/pi/birdclass/birds.gif')
    web_stream.occurrences([('Robin', '05/07/2022 14:52:00'), ('Robin', '05/07/2022 15:31:00')])
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
