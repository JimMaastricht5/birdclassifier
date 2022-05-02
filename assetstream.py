#!/usr/bin/env python3
import multiprocessing
import datetime


class AssetStream:
    def __init__(self, queue):
        self.queue = queue

    def write_asset(self):
        while True:
            self.queue.put(datetime.datetime.now())


def main():
    queue = multiprocessing.Queue()
    asset_stream = AssetStream(queue=queue)
    p_write_asset = multiprocessing.Process(target=asset_stream.write_asset(), args=(), daemon=True)
    p_write_asset.start()
    p_write_asset.join()
    # p.terminate(), p.is_alive()


if __name__ == '__main__':
    main()
