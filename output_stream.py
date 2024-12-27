# MIT License
#
# 2024 Jim Maastricht
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This file consists of two classes that use multiprocessing to send data to the cloud
# The controller class handle sending messages, images and files to a second class
# called the WebStream.  The WebStreams job is to write the files to the cloud and do
# it in a distinct process so as not to block the detection of birds by the main app
# this is probably overkill, but it was a good learning exercise
#
# Note: the name of the file to send to the cloud would need to change for each feeder to prevent overwriting contents!
# import multiprocessing
import queue
import threading
import pandas as pd
from datetime import datetime
import os
# import gcs


class WebStream:
    # Receives requests from the controller and writes out the contents periodically
    # message format is expected to be a tuple with
    # event_num: unique int key for event
    # types: end_process, motion, prediction, prediction_final, flush, message
    #   end_process or None: stops the process and free the creator's block on join()
    #   motion, prediction, prediction_final, message: all contain text for the website to display
    #   flush: writes the current set of in memory content to a file for website display
    # Date time: string
    # Prediction: string
    # Image_name: string, image name on disk
    def __init__(self, t_queue, path: str = os.getcwd(), gcs_obj = None, caller_id: str = "default",
                 run_local: bool = False, debug: bool = False) -> None:
        """
        set up web stream class, load from csv files to see if this restart was the result of a crash and
        load saved data
        :param t_queue: multiprocessing queue to pull messages from
        :param path: str contain os path to working dir, writes out csv file to path to temporarily accumulate data
        :param gcs_obj: class to control reading and writing to gcs
        :param caller_id: caller identity, can be anything
        :param run_local: stops writing to web, prevents overwriting for testing or running w/o network access
        :param debug: true prints extra messages to console
        """
        self.queue = t_queue
        self.path = path + '/assets'
        print(self.path)
        self.df_list = []
        self.id = caller_id
        self.run_local = run_local
        self.debug = debug
        self.storage = gcs_obj
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
        return

    def request_handler(self) -> None:
        """
        call back function for the controller to send requests / messages to
        grab message, check type, and process appropriately.
        message is in this format:
        item = [feeder_name, event_num, msg_type, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), message,
                image_name]
        :return: None
        """
        item = []
        try:
            while True:
                item = self.queue.get()  # get the next item in the queue to write to disk
                if item is None:  # poison pill, end the process
                    return  # end process

                print(item)  # send item to console and log for debugging
                msg_type = item[2]  # message type is the 3 rd item the list, count from 0
                if msg_type == 'flush':  # event type is flush, write out the data to a file and to the cloud
                    self.df = pd.DataFrame(self.df_list,
                                           columns=['Feeder Name', 'Event Num', 'Message Type', 'Date Time',
                                                    'Message', 'Image Name'])
                    self.df.to_csv(f'{self.path}/webstream.csv')
                    if self.run_local is False and self.storage is not None:  # avoid overwriting web contents when testing
                        self.storage.send_file(name=f'{datetime.now().strftime("%Y-%m-%d")}webstream.csv',
                                               file_loc_name=f'{self.path}/webstream.csv')
                    else:
                        print(f'skipped flush of contents to cloud for testing. contents were \n {self.df.to_string()}')
                elif msg_type == 'occurrences':
                    if len(item[4]) > 0:  # in this case the message is a list and must be longer than 0
                        self.df_occurrences = pd.DataFrame(item[4], columns=['Species', 'Date Time'])  # in msg pos
                        self.df_occurrences.insert(0, "Feeder Name", "")
                        self.df_occurrences['Feeder Name'] = self.id
                        self.df_occurrences.to_csv(f'{self.path}/web_occurrences.csv')  # species, date time
                        if self.run_local is False and self.storage is not None:  # don't write to cloud when debugging
                            self.storage.send_file(name=f'{datetime.now().strftime("%Y-%m-%d")}web_occurrences.csv',
                                                   file_loc_name=f'{self.path}/web_occurrences.csv')
                        else:
                            print(f'skipped writing occurrences for testing. '
                                  f'contents were \n{self.df_occurrences.to_string()}')
                    else:
                        pass  # empty message
                else:  # basic message or other event type: message, match,motion, spotted, inconclusive, weather, ....
                    if len(item) == 6:  # list should be six items long to append to df if not write an error to log
                        self.df_list.append(item)
                    else:
                        print(f'error on item list size {len(item)}, with values {item}')
        except Exception as e:
            print('failed process item in queue.  item: ', item)
            print(e)
        return


# start of function to prevent memory sharing between processes
# def web_stream_worker(queue, path: str, caller_id: str, run_local: bool, debug: bool) -> None:
#     """
#     This function allows for the setup of the multiprocessing consumer outside the memory of the controller
#     The two processes cannot share memory or access one another directly and this func eliminates that
#     :param queue: multiprocessing queue to communicate back and forth
#     :param path: path for files such as images
#     :param caller_id: name of caller, or the name of the bird feeder
#     :param run_local: true if debugging do not write to cloud, overwrites web activity
#     :param debug: extra print if true
#     :return: none
#     """
#     web_stream = WebStream(queue=queue, path=path, caller_id=caller_id, run_local=run_local, debug=debug)
#     web_stream.request_handler()
#     return


class Controller:
    """
    Multiprocessing controller, send messages to WebStream to process when the CPU has a moment
    """
    def __init__(self, caller_id: str = "default", gcs_obj=None, run_local: bool = False,
                 debug: bool = False) -> None:
        """
        Set up class, uses a dataframe to store the content
        :param caller_id: name of the sender, can be anything, non-unique identifier
        :param gcs_obj: obj that handles reading and writing to and from gcs
        :param run_local: boolean defaults to false, prevents unintentional send to cloud for contents when testing
        :param debug: prints extra messages to console if true
        """
        self.path = os.getcwd()
        # self.queue = multiprocessing.Queue()
        self.queue = queue.Queue()
        self.gcs_obj = gcs_obj
        self.web_stream = WebStream(t_queue=self.queue, caller_id=caller_id, gcs_obj=gcs_obj,
                                    run_local=run_local, debug=debug)
        # self.p_web_stream = multiprocessing.Process(target=web_stream_worker, args=(self.queue, self.path, caller_id,
        #                                                                             run_local, debug), daemon=True)
        self.t_web_stream = threading.Thread(target=self.web_stream.request_handler, args=(), daemon=True)
        self.last_event_num = 0
        self.id = caller_id  # id name or number of sender
        self.run_local = run_local
        self.debug = debug
        # This dataframe is not used in this class, here for a reference since this is how the stream handler writes CSV
        self.df = pd.DataFrame({
                           'Feeder Name': pd.Series(dtype='str'),
                           'Event Num': pd.Series(dtype='int'),
                           'Message Type': pd.Series(dtype='str'),
                           'Date Time': pd.Series(dtype='str'),
                           'Message': pd.Series(dtype='str'),
                           'Image Name': pd.Series(dtype='str')})
        return

    def start_stream(self) -> None:
        """
        start the stream, message queue is active
        :return: none
        """
        # self.p_web_stream.start()
        print('in start_stream')
        self.t_web_stream.start()
        return

    def message(self, message: str, feeder_name: str = '', event_num: int = 0, msg_type: str = 'message',
                image_name: str = '', flush: bool = False) -> None:
        """
        send a message from the controller to the message stream processor
        :param message: string containing the message (type message) or values to log as a list  in the format
            ['Feeder Name', 'Event Num', 'Message Type', 'Date Time', 'Message', 'Image Name'] (type occurrence)
        :param feeder_name: name of the bird feeder sending the info
        :param event_num: incrementing count of events recorded
        :param msg_type: string of types "flush", "occurrences", "match", "message" with "message" as the default
        :param image_name: fully qualified path to file and file name
        :param flush: tells the handler to send the results to the cloud and disk
        :return: none
        """
        event_num = self.last_event_num if event_num == 0 else event_num
        feeder_name = self.id if feeder_name == '' else feeder_name
        item = [feeder_name, event_num, msg_type, datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), message,
                image_name]
        self.queue.put(item)
        if flush:
            self.flush()
        self.last_event_num = event_num
        if self.debug:
            print(f'output_stream.py message: write item to file {item}')
        return

    def occurrences(self, occurrence_list: list) -> None:
        """
        takes list of occurrences and puts it on the queue for processing
        :param occurrence_list:
        :return: none
        """
        item = [self.id, 0, 'occurrences', datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), occurrence_list]
        self.queue.put(item)
        return

    def flush(self) -> None:
        """
        put the flush command on the queue to write the contents of activities to the cloud
        :return: None
        """
        item = ['', 0, 'flush', datetime.now().strftime("%H:%M:%S"), '', '']
        self.queue.put(item)
        return

    def end_stream(self) -> None:
        """
        Close down the stream for the day. Flushes the queue, sends the shutdown command (none), and
        waits for the process to end
        :return: None
        """
        self.flush()  # write any pending contents to disk
        try:
            self.queue.put(None)  # transmit poison pill to stop child process
            # does not seem to be ending correctly ******
            # if self.p_web_stream.is_alive():  # wait for the child process to finish if it is still alive
            if self.t_web_stream.is_alive():  # wait for the child process to finish if it is still alive
                print('waiting for web stream to finish processing queue....')
                # self.p_web_stream.join(timeout=30)
                self.t_web_stream.join(timeout=30)
        except Exception as e:
            print('attempted to end stream and failed')
            print(e)
        finally:
            print('Stream ended')
        return


# test code
def main():
    ctl = Controller(caller_id='local testing', run_local=True, debug=True)
    ctl.start_stream()
    print('starting stream')
    ctl.message('up and running')  # place message on queue for child process
    print('sent up and running msg')
    ctl.message(feeder_name='1SP', event_num=1, msg_type='prediction', message='big fat robin 97.0%',
                image_name='/home/pi/birdclass/first_img.jpg')
    ctl.message(feeder_name='1SP', event_num=1, msg_type='prediction', message='big fat robin 37.0%',
                image_name='/home/pi/birdclass/first_img.jpg')
    ctl.message(feeder_name='1SP', event_num=1, msg_type='prediction', message='big fat robin 96.0%',
                image_name='/home/pi/birdclass/first_img.jpg')
    ctl.message(feeder_name='1SP', event_num=1, msg_type='final_prediction', message='big fat robin 74.0%',
                image_name='/home/pi/birdclass/birds.gif')
    ctl.occurrences([('Robin', '05/07/2022 14:52:00'), ('Robin', '05/07/2022 15:31:00')])
    print('finished test code waiting for end stream')
    ctl.end_stream()


if __name__ == '__main__':
    main()
