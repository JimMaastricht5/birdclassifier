# MIT License
#
# 2021 Jim Maastricht
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
# auth.py must be located in project; protect this file as it contains keys
# code by JimMaastricht5@gmail.com
from datetime import datetime
from threads_api.src.threads_api import ThreadsAPI
import asyncio
from auth import (
    meta_user,
    meta_pwd
)

class MetaThreads:
    def __init__(self):
        self.meta_threads_api = ThreadsAPI()  # setup meta threads social app api
        return
    async def meta_threads_login(self):
        is_success = await self.meta_threads_api.login(username=meta_user, password=meta_pwd)
        print(f'login status: {is_success}')
        return

    async def post_status(self, message):
        result = await self.meta_threads_api.post(message)
        if result.media.pk:
            print(f'post successful')
        else:
            print(f'unable to post')
        return

    async def post_include_image_from_url(self, message, image_url):
        result = await self.meta_threads_api.post(message, image_path=image_url)
        if result.media.pk:
            print(f"Post has been successfully posted with id: [{result.media.pk}]")
        else:
            print("Unable to post.")
        return

async def meta_threads_login(meta_threads_api):
    is_success = await meta_threads_api.login(username=meta_user, password=meta_pwd)
    print(f'login status: {is_success}')
    return

async def post_status(meta_threads_api, message):
    result = await meta_threads_api.post(message)
    if result.media.pk:
        print(f'post successful')
    else:
        print(f'unable to post')
    return

async def post_include_image_from_url(meta_threads_api, message, image_url):
    result = await meta_threads_api.post(message, image_path=image_url)
    if result.media.pk:
        print(f"Post has been successfully posted with id: [{result.media.pk}]")
    else:
        print("Unable to post.")

async def post_include_image(meta_threads_api, message, image_path):
    result = await meta_threads_api.post(message, image_path=image_path)
    if result.media.pk:
        print(f"Post has been successfully posted with id: [{result.media.pk}]")
    else:
        print("Unable to post.")

async def main_test_meta():
    # threads-api  cryptography instagrapi packages required
    # meta_threads = MetaThreads()
    meta_threads_api = ThreadsAPI()
    await meta_threads_login(meta_threads_api)
    # test code to post a message
    message = f'Python status {datetime.now()}'
    # await post_status(meta_threads_api,message)
    # await post_include_image_from_url(meta_threads_api, message, 'https://storage.googleapis.com/tweeterssp-web-site-contents/2023-09-16-15-15-321359(HouseFinch).gif')
    await post_include_image(meta_threads_api, message, 'archive/birds.gif')
    await meta_threads_api.close_gracefully()


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_test_meta())
