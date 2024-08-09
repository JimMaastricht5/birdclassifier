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
# collection of image enhancement and processing techniques along with testing code
# many of the functions that are here are simple wrappers for Pillow functions.  Easier to remember this way
# from PIL import ImageStat
from PIL import ImageEnhance, Image, ImageOps, ImageFilter, ImageChops
import numpy as np
import io
import os


def grayscale(img: Image.Image) -> Image.Image:
    """
    create a gray scale image, used in motion detector to simplify image comparison
    :param img: RGB color jpg img
    :return: img
    """
    return ImageOps.grayscale(img)


def gaussianblur(img: Image.Image) -> Image.Image:
    """
    blur the image, used in motion detector to enhance contours of objects
    :param img: jpg
    :return: img
    """
    img.filter(ImageFilter.GaussianBlur)
    return img


# detect image problems where bottom half of the image is washed from suns reflection, must be jpg
def is_sun_reflection_jpg(img: Image.Image, washout_red_threshold: float = .25) -> bool:
    """
    function looks at an image and determines if it is overexposed.  On current hardware that results in
    the bottom half of the image having a pink or red hue.  Do the test is color diff from top to bottom
    :param img: jpg or gif to test for over exposure aka washout
    :param washout_red_threshold: threshold that is the % change from the bottom of the image in red spectrum
    :return: bool true if the image is over exposed
    """
    if img.format == 'GIF':
        first_frame = img.copy().convert('RGB')  # copy the img as RGB so we have three channels
        img.seek(1)  # get to first frame
        img_np_array = np.array(first_frame)
    else:
        img_np_array = np.array(img)

    if img_np_array.shape[2] == 3:  # is the 3rd dimension 3 for RGB
        height, width, channel = img_np_array.shape  # Get image dimensions
        top_half = img_np_array[:height // 2, :]  # Split the image into top and bottom halves
        bottom_half = img_np_array[height // 2:, :]
        # Calculate average red intensity for each half, washed out images are pink on the bottom half
        top_red_avg = np.mean(top_half[:, :, 0])
        bottom_red_avg = np.mean(bottom_half[:, :, 0])
        reflection_b = True if bottom_red_avg > top_red_avg * (1 + washout_red_threshold) else False
        print(f'top avg red is {top_red_avg}, bottom red avg is {bottom_red_avg} threshold is '
              f'{washout_red_threshold} with a limit of {top_red_avg * (1+washout_red_threshold)}')
    else:
        print(f'image_proc.py is_sun_reflection got np array with something other than 3 dimensions. '
              f'{img.format} with {img_np_array.shape}')
        reflection_b = False  # drop thru and return false if conversion or np dimensions does not return 3 channels
    return reflection_b


def resize(img: Image.Image, new_height: int, new_width: int, maintain_aspect: bool = True,
           box: tuple = None, resample: int = None) -> Image.Image:
    """
    function to resize image.  Full image or subset
    box is the subset of the image to perform an op on, default is entire img
    :param img: Pillow jpg img to process
    :param new_height: int with new height of image
    :param new_width: int with new width of image
    :param maintain_aspect: true or false, default is true
    :param box: tuple in the format of (starting_width, starting_height, ending_width, ending_height)
    :param resample: Image.BICUBIC(3) default.  Use LANCZOS(1) for improved down sampling, slower, BILINEAR (2) fastest
    :return: img
    """
    width, height = img.size
    new_height = int(height * (new_width / width)) if maintain_aspect else new_height
    img = img.resize((new_width, new_height), resample=resample, box=box)
    return img


def enhance(img: Image.Image, brightness: float = 1.0, sharpness: float = 1.0, contrast: float = 1.0,
            color: float = 1.0) -> Image.Image:
    """
    img is passed in along with brightness, sharpness, contrast and color changes
    1.0 is the default value and is no change
    decimal values can be used to reduce or increase an item
    .8 brightness would reduce the image brightness by 20%
    recommended values .8 to 1.2
    :param img: jpg
    :param brightness: adjust brightness
    :param sharpness: adjust sharpness
    :param contrast: adjust contrast
    :param color: adjust color
    :return: enhanced image
    """
    img = ImageEnhance.Color(img).enhance(brightness) if brightness != 1.0 else img
    img = ImageEnhance.Sharpness(img).enhance(sharpness) if sharpness != 1.0 else img
    img = ImageEnhance.Contrast(img).enhance(contrast) if contrast != 1.0 else img
    img = ImageEnhance.Color(img).enhance(color) if color != 1.0 else img
    return img


def area(rect: tuple) -> float:
    """
    :param rect: tuple with starting x, y, and ending x and y
    :return: float with area of rectangle
    """
    # find area of an image, used by overlap area
    (startX, startY, endX, endY) = rect
    return abs(endX - startX) * abs(endY - startY)


def overlap_area(rect1: tuple, rect2: tuple) -> float:
    """
    find the % area in 2 rectangles that overlaps, used to understand how much of a predicted box overlaps
    another on an image
    :param rect1: tuple starting x, y and ending x, y
    :param rect2: tuple starting x, y and ending x, y
    :return: float containing ratio of overlap 0 to 1.0
    """
    # find the % area in 2 rectangles that overlap
    (XA1, YA1, XA2, YA2) = rect1
    (XB1, YB1, XB2, YB2) = rect2
    sa = area(rect1)
    sb = area(rect2)
    si = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
    su = sa + sb - si
    return si/su


def compare_images(img_c1: Image.Image, img_c2: Image.Image) -> Image.Image:
    """
    compare two PIL images for differences
    :param img_c1: img 1
    :param img_c2: img 2
    :return: returns an img that is the brightness differences in each channel
    """
    if np.array(img_c1).shape != np.array(img_c2).shape:
        raise Exception(f'images are not the same shape img1:{np.array(img_c1).shape}, img2:{np.array(img_c2).shape}')
    return ImageChops.difference(img_c2, img_c1)


def convert_image(img: Image.Image) -> Image.Image:
    """
    convert a jpb image to a gif
    img: jpg img
    :return: gif image
    """
    stream = io.BytesIO()
    img.save(stream, 'gif')
    stream.seek(0)
    new_img = Image.open(stream)
    return new_img


def avg_exposure(img: Image.Image) -> float:
    """
    determine the avg exposure for an image
    :param img: jpg
    :return: float with average for entire image
    """
    return float(np.mean(np.array(img)))


def normalize(img: Image.Image) -> np.array:
    """
    normalize a jpg with values from 0 to 1 by dividing by max value of 255
    :param img: jpg img
    :return: np array with scaled image data 0 to 1
    """
    return np.array(img, dtype=np.float32) / 255.0


def crop(img: Image.Image, rect: tuple[int, int, int, int]) -> Image.Image:
    """
    crop an image from a rect passed as a tuple
    :param img: pillow image to crop
    :param rect: crop area as a tuple containing startx, starty, endx, endy
    :return: cropped image
    """
    return img.crop(rect)


def pad(img: Image.Image, new_width: int, new_height: int) -> Image.Image:
    """
    add black pixels to an image.  this is useful to get a cropped object back to the corerct size for a model
    :param img:
    :param new_width: new width of img
    :param new_height: new height of img
    :return:
    """
    if img.size[0] > new_width or img.size[1] > new_height:
        print(f'WARNING: Imageproc.py pad function is being asked to pad an image of size {img.size},'
              f'but new width and height are smaller ({new_width},{new_height}). Performing pad operation....')
    padded_img = Image.new('RGB', (new_width, new_height))  # create larger all black image
    padded_img.paste(img)  # paste org image over black pad at 0, 0 which is the upper left
    return padded_img


def save_gif(frames: list, frame_rate: int = 30,
             filename: str = os.getcwd()+'/assets/birds.gif') -> tuple[Image.Image, str]:
    """
    takes a list of jpg images and converts them to an animated gif
    :param frames: list of jpgs
    :param frame_rate: used to calc time in ml seconds per frame displayed
    :param filename: filename to write gif out
    :return: animated gif and the name of the file
    """
    gif_frames = [convert_image(frame) for frame in frames]
    try:
        gif_frame_one = gif_frames[0]  # grab a frame to save full image with
        ml_sec = int(1000 * len(gif_frames) * 1/frame_rate)  # frames * rate, 200 * 1/30 = 5 sec * 1,000 = ml sec
        gif_frame_one.save(filename, format="GIF", append_images=gif_frames[1:],
                           save_all=True, optimze=True, minimize_size=True, duration=ml_sec, loop=50)  # loop=0 infinity
        gif = open(filename, 'rb')  # reload gif
    except Exception as e:
        print(e)
        gif = frames[0]
    return gif, filename


# invoke main
if __name__ == "__main__":
    # print(overlap_area((1, 1, 10, 10), (1, 1, 2, 2)))
    img1 = Image.open('/home/pi/birdclass/birds.gif')
    # img = Image.open('/home/pi/birdclass/washout3.jpg')
    print(img1.format)
    # print(avg_exposure(img1))
    # print(is_sun_reflection_jpg(img1))
    # img1 = crop(img1, (100, 100, 200, 300))
    img1 = pad(img1, 100, 100)
    img1.show()

    # img2 = Image.open('/home/pi/birdclass/washout6.gif')
    # print(img2.format)
    # print(avg_exposure(img2))
    # print(is_sun_reflection_jpg(img2))
    # img2.show()

    # img = resize(img_org, 100, 100, maintain_aspect=False)
    # gif1 = convert_image(img1, target='gif', save_test_img=True)
    # img2 = Image.open('/home/pi/birdclass/test2.jpg')
    # img2_gif = convert_image(img=img2, target='gif')
    # img3_gif = convert_image(img=img2, target='gif')
    # img3_gif.save('/home/pi/birdclass/test stream.gif', 'gif')
    # gif2 = convert_image(img2, target='gif', save_test_img=True)
    # save_gif([img1, img2], frame_rate=10, filename='/home/pi/birdclass/test4.gif')

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


# old code
# def contour(img):
#     # find contours of the image
#     img.filter(ImageFilter.CONTOUR)
#     return img
# def enhance_color(img, factor):
#     # color enhance image
#     # factor of 1 is no change. < 1 reduces color,  > 1 increases color
#     # recommended values for color pop of 1.2
#     # recommended values for reductions 0.8
#     return ImageEnhance.Color(img).enhance(factor)
#
#
# def enhance_brightness(img, factor):
#     # brighten or darken an image
#     # factor of 1 is no change. < 1 reduces color,  > 1 increases color
#     # recommended values of 1.2 or 0.8
#     return ImageEnhance.Brightness(img).enhance(factor)
#
#
# def enhance_contrast(img, factor):
#     # increases or decreases contrast
#     # factor of 1 is no change. < 1 reduces color,  > 1 increases color
#     # recommended values 1.5, 3, 0.8
#     return ImageEnhance.Contrast(img).enhance(factor)
#
#
# def enhance_sharpness(img, factor):
#     # increases or decreases sharpness
#     # factor of 1 is no change. < 1 reduces color,  > 1 increases color
#     # recommended values 1.5, 3
#     # use 0.2 for blur
#     return ImageEnhance.Sharpness(img).enhance(factor)


# def is_color_low_contrast(colorimg, threshold=.35):
#     # check in image to see if is low contrast, return True or False
#     # input image and threshold as a decimal with .35 or 35% being the default
#     stats = ImageStat.Stat(colorimg)
#     if stats.stdev < threshold:
#         return False
#     else:
#         return True


# def equalize_gray(grayimg):
#     # adjust contrast of gray image to improve process
#     # apply histogram equalization to boost contrast
#     return ImageOps.equalize(grayimg)


# def equalize_color(img):
#     # color histogram equalization
#     return ImageOps.equalize(img)


# def ratio(rect):
#     # find the ratio of width/height
#     (startX, startY, endX, endY) = rect
#     return round((endX - startX) / (endY - startY), 3)


# def flip(img):
#     # Pillow img to flip
#     return ImageOps.flip(img)
