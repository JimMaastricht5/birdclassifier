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
from skimage.exposure import is_low_contrast


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
    if isinstance(img, (Image.Image)) and convert_to == 'np':
        img = np.array(img)
    elif isinstance(img, (np.ndarray)) and convert_to == 'PIL':
        img = Image.fromarray(img)
    # else requires no conversion
    return img


# enhance color, brightness, and contrast
# provides all three stages back for viewing
# take either a pil image or nparray and returns nparray
def enhancements(img):
    img = convert(img, 'PIL')  # converts to PIL format if necessary
    img_clr = enhance_color(img, 1.0)
    img_clr_brt = enhance_brightness(img_clr, 1.2)
    img_clr_brt_con = enhance_contrast(img_clr_brt, 1.0)
    return convert(img_clr), convert(img_clr_brt), convert(img_clr_brt_con)  # return numpy arrays


# color enhance image
# factor of 1 is no change. < 1 reduces color,  > 1 increases color
# recommended values for color pop of 1.5 or 3
# recommended values for reductions 0.8, 0.4
# B&W is 0
def enhance_color(img, factor):
    return ImageEnhance.Color(img).enhance(factor)


# brighten or darken an image
# factor of 1 is no change. < 1 reduces color,  > 1 increases color
# recommended values for brightness of 1.2 or 0.8
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


# estimate the size of the bird based on the percentage of image area consumed by the bounding box
def objectsize(args, startx, starty, endx, endy):
    objarea = abs((startx - endx) * (starty - endy))
    scrarea = args['screenheight'] * args['screenwidth']
    perarea = (objarea / scrarea) * 100
    if perarea >= 40:  # large
        size = 'L'
    elif perarea >= 30:  # medium
        size = 'M'
    else:  # small
        size = 'S'
    return size, perarea


# compare two PIL images for differences
# returns an array of the differences
def compare_images(img1, img2):
    return ImageChops.difference(img2, img1)


# compare two gray scale images
# https://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images
def compare_images2(img1, img2):
    # normalize to compensate for exposure difference
    img1 = equalize_gray(img1)
    img2 = equalize_gray(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    # z_norm = norm(diff.ravel(), 0)  # Zero norm ravel from scipy?
    return m_norm


# testing code
def main():
    imgnp = Image.open('/home/pi/birdclass/test2.jpg')
    imgnp.show()

    # test image bad contrast and equalization
    grayimg = ImageOps.grayscale(imgnp)
    print(is_low_contrast(grayimg))
    equalizedimg1 = equalize_gray(grayimg)
    equalizedimg1.show()

    # color equalize
    equalizedcolorimg = equalize_color(imgnp)
    equalizedcolorimg.show()
    print(predominant_color(equalizedcolorimg))


# invoke main
if __name__ == "__main__":
    main()
