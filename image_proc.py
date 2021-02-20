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

from PIL import ImageEnhance, Image
import cv2
import numpy as np
from skimage.exposure import is_low_contrast


# testing code
def main():
    imgnp = cv2.imread('/home/pi/birdclass/test2.jpg')
    cv2.imshow('org', imgnp)  # show orginal image

    # test image bad contrast and equalization
    grayimg = cv2.cvtColor(imgnp, cv2.COLOR_BGR2GRAY)
    print(is_low_contrast(grayimg))
    equalizedimg1 = equalize_gray(grayimg, type=1)
    equalizedimg2 = equalize_gray(grayimg, type=2)
    cv2.imshow('equalized simple 1', equalizedimg1)
    cv2.imshow('equalized simple 2', equalizedimg2)

    # color equalize
    equalizedcolorimg = equalize_color(imgnp)
    cv2.imshow('color histogram equalization', equalizedcolorimg)
    # convert image to pil, test scheme to adjust pil or np images
    # imgpil = Image.fromarray(cv2.imread('/home/pi/birdclass/test2.jpg'))  # need to pass a PIL Image vs. Numpy array
    # img_clr, img_clr_brt, img_clr_brt_con = enhancements(imgnp)
    # img_clr, img_clr_brt, img_clr_brt_con = enhancements(imgpil)
    # cv2.imshow('color', img_clr)
    # cv2.imshow('color brt', img_clr_brt)
    # cv2.imshow('color brt con', img_clr_brt_con)

    # check for esc key and quit if pressed
    while True:
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break


# pillow requires a pillow image to work
# func provides both formats for conversion
# default conversion is to numpy array
def convert(img, convert_to = 'np'):
    if isinstance(img,(Image.Image)) and convert_to == 'np':
        img = np.array(img)
    elif isinstance(img,(np.ndarray)) and convert_to == 'PIL':
        img = Image.fromarray(img)
    # else requires no conversion
    return img


# enhance color, brightness, and contrast
# provides all three stages back for viewing
# take either a pil image or nparray and returns nparray
def enhancements(img):
    img = convert(img, 'PIL')  # converts to PIL format if necessary
    img_clr = enhance_color(img, 2.5)
    img_clr_brt = enhance_brightness(img_clr, 1.0)
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
# uses scikit-images is_low_contrast func on gray scale image
# input image and threshold as a decimal with .35 or 35% being the default
# takes an image as np or pil format
def low_contrast_detector(grayimg, threshold=.35):
    return is_low_contrast(grayimg, threshold)  # scikit image function


# adjust contrast of gray image to improve process
# apply histogram equalization to boost contrast
# uses type: simple (1) or adaptive (2) equalization
# calc intensity of gray scale image and spread equaly
def equalize_gray(grayimg, type=2):
    if type == 1:
        equalizedgrayimg = cv2.equalizeHist(grayimg)
    else:
        # create clahe algorithm object, clipLimit is the threshold for contract limiting
        # tileGridsize divides the image in M x N tiles then applies the histogram
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize= (8,8))
        equalizedgrayimg = clahe.apply(grayimg)
    return equalizedgrayimg


# color histogram equalization
# no standard cv2 functions for color equalization.
# can be accomplished by splitting colors and manually adjusting each
# credit to....
# https://towardsdatascience.com/histogram-equalization-a-simple-way-to-improve-the-contrast-of-your-image-bcd66596d815
def equalize_color(img):
    # segregate color streams
    b, g, r = cv2.split(img)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')


    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')
    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
    return cv2.merge((img_b, img_g, img_r))


def predominant_color(img):
    # from pyimagesearch.com color detection
    # define the list of boundaries.  create sets of Green Blue Red GBR defining lower and upper bounds
    i = 0
    colorcount = [''] * 4
    colors = np.array(['Red', 'Blue', 'Yellow', 'Gray'])
    boundaries = [
        ([17, 15, 100], [50, 56, 200]),  # Red
        ([86, 31, 4], [220, 88, 50]),  # Blue
        ([25, 146, 190], [62, 174, 250]),  # Yellow
        ([103, 86, 65], [145, 133, 128])  # Gray
    ]

    # grab background color for initial mask
    # bkgcolor = int(img[0, 0])  # bgr value in upper left corner
    # bkgmask = cv2.inRange(img, bkgcolor, ...)

    # loop over the boundaries
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")  # create NumPy arrays from the boundaries
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(img, lower, upper)
        maskimg = cv2.bitwise_and(img, img, mask=mask)
        colorcount[i] = np.count_nonzero(maskimg, axis=None)  # count non-zero values in BGR pixels
        i += 1

    cindex = np.where(colorcount == np.amax(colorcount))  # find color with highest count
    color = str(colors[cindex])
    return color


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


# invoke main
if __name__ == "__main__":
    main()