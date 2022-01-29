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
#
# lib of image enhancement and processing techniques
# required Pillow, scikit-image library in project
from PIL import ImageEnhance, Image, ImageOps, ImageStat, ImageFilter, ImageChops
import numpy as np
import io
# from skimage.exposure import is_low_contrast


# Pillow img to flip
def flip(img):
    return ImageOps.flip(img)


# create a gray scale image
def grayscale(img):
    return ImageOps.grayscale(img)


# blur the image
def gaussianblur(img):
    img.filter(ImageFilter.GaussianBlur)
    return img


# find countours of the image
def contour(img):
    img.filter(ImageFilter.CONTOUR)
    return img


# func provides both formats for conversion
# default conversion is to numpy array
def convert(img, convert_to='np'):
    if isinstance(img, Image.Image) and convert_to == 'np':
        img = np.array(img)
    elif isinstance(img, np.ndarray) and convert_to == 'PIL':
        img = Image.fromarray(img)
    # else requires no conversion
    return img


# enhance color, brightness, sharpness, and contrast
# take either a pil image
def enhance(img, brightness=1.0, sharpness=1.0, contrast=1.0, color=1.0):
    img = enhance_brightness(img, brightness)
    img = enhance_sharpness(img, sharpness)
    img = enhance_contrast(img, contrast)
    img = enhance_color(img, color)
    return img


# color enhance image
# factor of 1 is no change. < 1 reduces color,  > 1 increases color
# recommended values for color pop of 1.2
# recommended values for reductions 0.8
def enhance_color(img, factor):
    return ImageEnhance.Color(img).enhance(factor)


# brighten or darken an image
# factor of 1 is no change. < 1 reduces color,  > 1 increases color
# recommended values of 1.2 or 0.8
def enhance_brightness(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)


# increases or decreases contrast
# factor of 1 is no change. < 1 reduces color,  > 1 increases color
# recommended values 1.5, 3, 0.8
def enhance_contrast(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)


# increases or decreases sharpness
# factor of 1 is no change. < 1 reduces color,  > 1 increases color
# recommended values 1.5, 3
# use 0.2 for blur
def enhance_sharpness(img, factor):
    return ImageEnhance.Sharpness(img).enhance(factor)


# check in image to see if is low contrast, return True or False
# input image and threshold as a decimal with .35 or 35% being the default
# takes an image as pil format
def is_color_low_contrast(colorimg, threshold=.35):
    stats = ImageStat.Stat(colorimg)
    if stats.stdev < threshold:
        return False
    else:
        return True


# adjust contrast of gray image to improve process
# apply histogram equalization to boost contrast
def equalize_gray(grayimg):
    return ImageOps.equalize(grayimg)


# color histogram equalization
def equalize_color(img):
    return ImageOps.equalize(img)


# from pyimagesearch.com color detection
def predominant_color(pil_img):
    img = pil_img.copy()
    img.convert("RGB")
    img.resize((1, 1), resample=0)
    dominant_color = img.getpixel((0, 0))
    return dominant_color


# find area of an image
def area(rect):
    (startX, startY, endX, endY) = rect
    return abs(endX - startX) * abs(endY - startY)


# find the % area in 2 rectangles that overlap
def overlap_area(rect1, rect2):
    (startX, startY, endX, endY) = rect1
    area1 = area(rect1)
    (startX2, startY2, endX2, endY2) = rect2
    area2 = area(rect2)
    x_dist = min(endX, endX2) - max(startX, startX2)
    y_dist = min(endY, endY2) - max(startY, startY2)
    if x_dist > 0 and y_dist > 0:
        area_i = x_dist * y_dist
    else:
        area_i = 0
    return (area1 + area2 - area_i) / (area1 + area2)


# compare two PIL images for differences
# returns an array of the differences
def compare_images(img1, img2):
    if np.array(img1).shape != np.array(img2).shape:
        raise Exception(f'images are not the same shape img1:{np.array(img1).shape}, img2:{np.array(img2).shape}')
    return ImageChops.difference(img2, img1)


def convert_image(img, target='gif'):
    stream = io.BytesIO()
    img.save(stream, target)
    stream.seek(0)
    new_img = Image.open(stream)
    return new_img


# takes list of frames and saves as a gif
def save_gif(frames, frame_rate=30, filename='/home/pi/birdclass/birds.gif'):
    gif_frames = [convert_image(frame, target='gif') for frame in frames]
    try:
        gif_frame_one = gif_frames[0]  # grab a frame to save full image with
        ml_sec = int(1000 * len(gif_frames) * 1/frame_rate)  # frames * rate, 200 * 1/30 = 5 sec * 1,000 = ml sec
        gif_frame_one.save(filename, format="GIF", append_images=gif_frames[1:],
                           save_all=True, optimze=True, minimize_size=True, duration=ml_sec, loop=50)  # loop=0 infinity
        gif = open(filename, 'rb')  # reload gif
    except Exception as e:
        print(e)
        gif = frames[0]
    return gif


# testing code
def main():
    # print(area((1, 1, 4, 4)))
    # print(overlap_area((1, 1, 4, 4), (1, 1, 2, 2)))
    img1 = Image.open('/home/pi/birdclass/test.gif')
    # gif1 = convert_image(img1, target='gif', save_test_img=True)
    img2 = Image.open('/home/pi/birdclass/test2.jpg')
    img2_gif = convert_image(img=img2, target='gif')
    img3_gif = convert_image(img=img2, target='gif')
    img3_gif.save('/home/pi/birdclass/test stream.gif', 'gif')
    # gif2 = convert_image(img2, target='gif', save_test_img=True)
    save_gif([img1, img2], frame_rate=10, filename='/home/pi/birdclass/test4.gif')
    # img.show()

    # img = enhance_brightness(img, 1)
    # img = enhance_sharpness(img, 1.2)
    # img = enhance_contrast(img, 1.2)
    # img = enhance_color(img, 1.2)
    # img = equalize_color(img)
    # img = enhance(img, brightness=1.3, sharpness=1.2, contrast=1.2, color=1.2)
    # img.show()

    # test image bad contrast and equalization
    # grayimg = ImageOps.grayscale(img)
    # print(is_low_contrast(grayimg))
    # equalizedimg1 = equalize_gray(grayimg)
    # equalizedimg1.show()
    #
    # # color equalize
    # equalizedcolorimg = equalize_color(img)
    # equalizedcolorimg.show()
    # print(predominant_color(equalizedcolorimg))


# invoke main
if __name__ == "__main__":
    main()
