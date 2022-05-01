#!/usr/bin/env python3
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import socketserver
import multiprocessing
import datetime
import time
import codecs


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


class HTTPHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        self.web_filename = '/home/pi/birdclass/index.html'
        self.page_head = """\
            <html>
            <head>
            <title>picamera MJPEG streaming demo</title>
            <meta http-equiv="refresh" content="1">
            </head>
            <body>
            <h1>PiCamera MJPEG Streaming Demo</h1>
            <p>
            """
        self.page_tail = """\
                    </p>
                    <img src="stream.mjpg" width="640" height="480" />
                    </body>
                    </html>
                    """
        BaseHTTPRequestHandler.__init__(self, request, client_address,
                                        server)  # super's init called after setting attribute

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        item = self.read_queue()
        message = item
        print(message)

        # with open('/home/pi/birdclass/index.html', 'r') as f:  # this should not be hardcoded
        #     message = f.read()
        #     # message = self.page_head + str(item[0]) + ':' + str(item[1]) + self.page_tail  # redundant
        if item is not None:
            self.wfile.write(bytes(message, "utf8"))

    def read_queue(self):
        item = None
        if self.server.queue is None:
            return item
        try:
            while True:
                item = self.server.queue.get_nowait()
                self.server.last_item = item
        except queue.Empty:
            pass
        return item


class WebServer:
    def __init__(self, queue, port=8080):
        self.port = port
        self.queue = queue
        self.last_item = (0, '')
        self.full_text = ''

    def start_threaded_server(self):
        server = ThreadedHTTPServer(('', self.port), HTTPHandler)
        server.queue = self.queue
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        print('HOST IP:', host_ip)
        socket_address = (host_ip, self.port)
        print("LISTENING AT:", socket_address)
        with server:
            server.serve_forever()


class WebControl:
    def __init__(self, web_filename='/home/pi/birdclass/index.html',
                 template_filename='/home/pi/birdclass/template.html',
                 gif_filename='/home/pi/birdclass/birds.gif'):
        self.queue = multiprocessing.Queue()
        self.web = WebServer(queue=queue, port=8080)
        self.p_web_server = multiprocessing.Process(target=self.web.start_threaded_server, args=(), daemon=True)
        self.p_web_server.start()
        self.web_file_name = web_filename
        self.gif_filename = gif_filename
        # read template file and reset target html file
        template_file = codecs.open(template_filename, 'r', "utf8")
        template_page = template_file.read()
        self.web_file = open(self.web_filename, 'w')  # open file to write msg and images into....
        self.web_file.write(template_page)
        template_file.close()

    def end_web(self):
        try:
            self.web_file.close()
            self.p_web_server.terminate()
        finally:
            return


# **** TESTING Code
# produce class for testing
class WidgetProducer:
    def __init__(self, queue):
        self.queue = queue
        self.item_num = 0
        self.template_filename = '/home/pi/birdclass/template.html'

    def produce(self):
        while True:
            with codecs.open(self.template_filename, 'r', "utf-8") as f:
                self.queue.put(f.read())
            time.sleep(1)
            self.item_num += 1
            # self.queue.put((self.item_num, datetime.datetime.now()))



def main():
    web_filename='/home/pi/birdclass/index.html'
    template_filename='/home/pi/birdclass/template.html'
    gif_filename='/home/pi/birdclass/birds.gif'
    # read template file and reset target html file
    template_file = codecs.open(template_filename, 'r', "utf-8")
    template_page = template_file.read()
    web_file = open(web_filename, 'w')  # open file to write msg and images into....
    web_file.write(template_page)
    template_file.close()
    web_file.close()

    queue = multiprocessing.Queue()
    producer = WidgetProducer(queue=queue)
    web = WebServer(queue=queue, port=8080)
    p_web_server = multiprocessing.Process(target=web.start_threaded_server, args=(), daemon=True)
    p_producer = multiprocessing.Process(target=producer.produce, args=(), daemon=True)
    p_web_server.start()
    p_producer.start()
    p_web_server.join()
    p_producer.join()
    # p.terminate(), p.is_alive()


if __name__ == '__main__':
    main()

# # class StreamingOutput(object):
# #     def __init__(self):
# #         self.frame = None
# #         self.buffer = io.BytesIO()
# #         self.condition = Condition()
# #
# #     def write(self, buf):
# #         if buf.startswith(b'\xff\xd8'):
# #             # New frame, copy the existing buffer's content and notify all
# #             # clients it's available
# #             self.buffer.truncate()
# #             with self.condition:
# #                 self.frame = self.buffer.getvalue()
# #                 self.condition.notify_all()
# #             self.buffer.seek(0)
# #         return self.buffer.write(buf)
# #
# #
# # class StreamingHandler(server.BaseHTTPRequestHandler):
# #     def do_GET(self):
# #         if self.path == '/':
# #             self.send_response(301)
# #             self.send_header('Location', '/index.html')
# #             self.end_headers()
# #         elif self.path == '/index.html':
# #             content = PAGE.encode('utf-8')
# #             self.send_response(200)
# #             self.send_header('Content-Type', 'text/html')
# #             self.send_header('Content-Length', len(content))
# #             self.end_headers()
# #             self.wfile.write(content)
# #         elif self.path == '/stream.mjpg' or self.path =='/stream.jpg':
# #             self.send_response(200)
# #             self.send_header('Age', 0)
# #             self.send_header('Cache-Control', 'no-cache, private')
# #             self.send_header('Pragma', 'no-cache')
# #             self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
# #             self.end_headers()
# #             try:
# #                 while True:
# #                     with output.condition:
# #                         output.condition.wait()
# #                         frame = output.frame
# #                     self.wfile.write(b'--FRAME\r\n')
# #                     self.send_header('Content-Type', 'image/jpeg')
# #                     self.send_header('Content-Length', len(frame))
# #                     self.end_headers()
# #                     self.wfile.write(frame)
# #                     self.wfile.write(b'\r\n')
# #             except Exception as e:
# #                 logging.warning(
# #                     'Removed streaming client %s: %s',
# #                     self.client_address, str(e))
# #         else:
# #             self.send_error(404)
# #             self.end_headers()
# #
# #
# # class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
# #     allow_reuse_address = True
# #     daemon_threads = True
# #
# #
# # def main():
# #     with picamera.PiCamera(resolution='640x480', framerate=24) as camera:
# #         output = StreamingOutput()
# #         camera.start_recording(output, format='jpg')
# #         try:
# #             address = ('', 8000)
# #             server = StreamingServer(address, StreamingHandler)
# #             server.serve_forever()
# #         finally:
# #             camera.stop_recording()
# #
# # # invoke main
# # if __name__ == "__main__":
# #     main()
# #
