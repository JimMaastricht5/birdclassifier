# lib of image enhancement and processing techniques
# required Pillow library in project

from PIL import ImageEnhance, Image
import cv2
import numpy as np


# testing code
def main():
    imgnp = cv2.imread('/home/pi/birdclass/test2.jpg')
    imgpil = Image.fromarray(cv2.imread('/home/pi/birdclass/test2.jpg'))  # need to pass a PIL Image vs. Numpy array
    img_clr, img_clr_brt, img_clr_brt_con = winter(imgnp)
    img_clr, img_clr_brt, img_clr_brt_con = winter(imgpil)
    cv2.imshow('org', imgnp)
    cv2.imshow('color', img_clr)
    cv2.imshow('color brt', img_clr_brt)
    cv2.imshow('color brt con', img_clr_brt_con)

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

# set of enhancements for winter weather in WI
# enhance color, brightness, and contrast
# provides all three stages back for viewing
# take either a pil image or nparray and returns nparray
def winter(img):
    img = convert(img, 'PIL')  # converts to PIL format if necessary
    img_clr = enhance_color(img, 2)
    img_clr_brt = enhance_brightness(img_clr, 1.2)
    img_clr_brt_con = enhance_contrast(img_clr_brt, 1.2)
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


if __name__ == "__main__":
    main()