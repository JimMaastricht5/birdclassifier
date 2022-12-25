#!/usr/bin/env python3
import multiprocessing
import pandas as pd
from datetime import datetime
import os
import gcs


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
    def __init__(self, queue, path=os.getcwd(), caller_id="default"):
        self.queue = queue
        self.path = path
        print(self.path)
        self.asset_path = self.path + '/assets'
        print(self.asset_path)
        self.df_list = []
        self.id = caller_id
        self.storage = gcs.Storage()
        # recover from crash without losing data.  Load data if present.  Keep if current, delete if yesterday
        try:
            self.df = pd.read_csv(f'{self.path}/webstream.csv')
            self.df_occurrences = pd.read_csv(f'{self.path}/web_occurrences.csv')
            df_date = pd.to_datetime(self.df.iloc[0]['Date Time'])
            print(f'Prior stream date from {df_date}, now: {datetime.now()}')
            if df_date < datetime.now():  # empty df if yesterday's data
                print('Emptying dataframe, data is stale')
                self.df.drop(self.df.index, inplace=True)
                self.df_occurrences.drop(self.df_occurrences.index, inplace=True)
        except FileNotFoundError:  # if no file was found build an empty df
            print('No prior stream file found, creating empty stream')
            self.df = pd.DataFrame({
                'Feeder Name': pd.Series(dtype='str'),
                'Event Num': pd.Series(dtype='int'),
                'Message Type': pd.Series(dtype='str'),
                'Date Time': pd.Series(dtype='str'),
                'Message': pd.Series(dtype='str'),
                'Image Name': pd.Series(dtype='str')})
            self.df_occurrences = pd.DataFrame({
                'Feeder Name': pd.Series(dtype='str'),
                'Species': pd.Series(dtype='str'),
                'Date Time': pd.Series(dtype='str')})
            pass

    def request_handler(self):
        item = []
        try:
            while True:
                item = self.queue.get()  # get the next item in the queue to write to disk
                if item is None:  # poison pill, end the process
                    return  # end process

                msg_type = item[2]  # message type is the 3 rd item the list, counjt from 0
                print('getting from q:', msg_type)
                if msg_type == 'flush':  # event type is flush
                    print('flush mem to disk and web', item)
                    print('list is', self.df_list)
                    self.df = pd.DataFrame(self.df_list,
                                           columns=['Feeder Name', 'Event Num', 'Message Type', 'Date Time',
                                                    'Message', 'Image Name'])
                    self.df.to_csv(f'{self.path}/webstream.csv')
                    self.storage.send_file(name=f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
                                                f'webstream.csv',
                                           file_loc_name=f'{self.path}/webstream.csv')
                elif msg_type == 'occurrences':
                    print('in occurrences', item)
                    if len(item[4]) > 0:  # list in a list in message position
                        print(item)  # send full array to console
                        self.df_occurrences = pd.DataFrame(item[4], columns=['Species', 'Date Time'])  # in msg pos
                        self.df_occurrences.insert(0, "Feeder Name", "")
                        self.df_occurrences['Feeder Name'] = self.id
                        print('sending file to disk and web....')
                        self.df_occurrences.to_csv(f'{self.path}/web_occurrences.csv')  # species, date time
                        self.storage.send_file(name=f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
                                                    f'web_occurrences.csv',
                                               file_loc_name=f'{self.path}/web_occurrences.csv')
                        print('return from file send to disk and web')
                    else:
                        pass  # empty message
                else:  # basic message or other event type: message, motion, spotted, inconclusive, weather, ....
                    print('in msg else', item)  # values may be missing so don't subscript here
                    if len(item) == 6:  # list should be six items long
                        self.df_list.append(item)
                    else:
                        print(f'error on item list size {len(item)}, with values {item}')
        except Exception as e:
            print('failed process item in queue.  item: ', item)
            print(e)
        return


class Controller:
    def __init__(self, caller_id="default"):
        self.queue = multiprocessing.Queue()
        self.web_stream = WebStream(queue=self.queue)
        self.p_web_stream = multiprocessing.Process(target=self.web_stream.request_handler, args=(), daemon=True)
        self.last_event_num = 0
        self.id = caller_id  # id name or number of sender
        self.df = pd.DataFrame({
                           'Feeder Name': pd.Series(dtype='str'),
                           'Event Num': pd.Series(dtype='int'),
                           'Message Type': pd.Series(dtype='str'),
                           'Date Time': pd.Series(dtype='str'),
                           'Message': pd.Series(dtype='str'),
                           'Image Name': pd.Series(dtype='str')})

    def start_stream(self):
        # self.p_web_stream.start(id=self.id)
        self.p_web_stream.start()
        return

    def message(self, message, feeder_name='', event_num=0, msg_type='message', image_name='', flush=False):
        # print('web controller sending: ', message)
        event_num = self.last_event_num if event_num == 0 else event_num
        feeder_name = self.id if feeder_name == '' else feeder_name
        item = [feeder_name, event_num, msg_type, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), message,
                image_name]
        # print('sending to queue', item)
        self.queue.put(item)
        if flush:
            self.flush()
        self.last_event_num = event_num
        return

    def occurrences(self, occurrence_list):
        # print(occurrence_list)
        # self.df = pd.DataFrame({
        #     'Feeder Name': pd.Series(dtype='str'),
        #     'Event Num': pd.Series(dtype='int'),
        #     'Message Type': pd.Series(dtype='str'),
        #     'Date Time': pd.Series(dtype='str'),
        #     'Message': pd.Series(dtype='str'),
        #     'Image Name': pd.Series(dtype='str')})
        item = [self.id, 0, 'occurrences', datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), occurrence_list]
        self.queue.put(item)
        return

    def flush(self):
        item = ['', 0, 'flush', datetime.now().strftime("%H:%M:%S"), '', '']
        self.queue.put(item)
        return

    def end_stream(self):
        self.flush()  # write any pending contents to disk
        try:
            self.queue.put(None)  # transmit poison pill to stop child process
            # does not seem to be ending correctly ******
            if self.p_web_stream.is_alive():  # wait for the child process to finish if it is still alive
                print('waiting for web stream to finish processing queue....')
                self.p_web_stream.join(timeout=30)
        except Exception as e:
            print('attempted to end stream and failed')
            print(e)
        finally:
            print('')
        return


def main():
    web_stream = Controller()
    web_stream.start_stream()
    web_stream.message('up and running')  # place message on queue for child process
    web_stream.message(feeder_name='1SP', event_num=1, msg_type='prediction', message='big fat robin 97.0%',
                       image_name='/home/pi/birdclass/first_img.jpg')
    web_stream.message(feeder_name='1SP', event_num=1, msg_type='prediction', message='big fat robin 37.0%',
                       image_name='/home/pi/birdclass/first_img.jpg')
    web_stream.message(feeder_name='1SP', event_num=1, msg_type='prediction', message='big fat robin 96.0%',
                       image_name='/home/pi/birdclass/first_img.jpg')
    web_stream.message(feeder_name='1SP', event_num=1, msg_type='final_prediction', message='big fat robin 74.0%',
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
