# lib of image enhancement and processing techniques
# required Pillow library in project

from PIL import ImageEnhance


# set of enhancements for winter weather in WI
# enhance color, brightness, and contrast
# provides all three stages back for viewing
def winter(img):
    img_clr = enhance_color(img, 3)
    img_clr_brt = enhance_brightness(img_clr, 1.2)
    img_clr_brt_con = enhance_contrast(img_clr_brt, 1.5)
    return img_clr, img_clr_brt, img_clr_brt_con


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



