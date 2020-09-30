# basic project info at  https://learn.adafruit.com/adafruits-raspberry-pi-lesson-4-gpio-setup/configuring-i2c
# modified to incorporate object tracking by JimMaastricht5@gmail.com
# packages: pan tilt uses PCA9685-Driver
from PCA9685 import PCA9685


# place detected object closer to the center of the camera's lens
def trackobject(pwm, cv, currpan, currtilt, img, objsdetected, screen_height, screen_width):
    panto = currpan
    tiltto = currtilt
    for (x, y, x2, y2) in objsdetected:
        # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        w = abs(x2 - x)
        h = abs(y2 - y)
        xfacecenter = x + w / 2
        yfacecenter = y + h / 2

        if abs(screen_width / 2 - xfacecenter) > 50:
            if xfacecenter > screen_width / 2:
                panto = currpan - 1
            else:
                panto = currpan + 1

        if abs(screen_height / 2 - yfacecenter) > 50:
            if yfacecenter > screen_height / 2:
                tiltto = currtilt + 2
            else:
                tiltto = currtilt - 2

        currpan, currtilt = setservoangle(pwm, currpan, panto, currtilt, tiltto)

    return currpan, currtilt


# set pan and tilt, move in the proper direction smoothly
def setservoangle(pwm, currpan, panto, currtilt, tiltto):

    tiltto = checkboundary(tiltto)
    panto = checkboundary(panto)
    tiltdir = setdirection(currtilt, tiltto)
    pandir = setdirection(currpan, panto)

    # move pan tilt smoothly by moving toward target to in increments of 1.
    # loop until both pan and tilt match target to setting
    while currpan != panto or currtilt != tiltto:
        pwm.setRotationAngle(0, currpan)
        pwm.setRotationAngle(1, currtilt)
        if currpan != panto:
            currpan += pandir  # if still need to pan; pan in + or - direction
        if currtilt != tiltto:
            currtilt += tiltdir  # if still need to tile; tilt in + or - direction

    return currpan, currtilt


# set direction for camera move.  calculate +1 or -1 using division and ABS
def setdirection(currpos, torequest):
    if torequest != currpos:
        movedirection = int((torequest - currpos) / abs(torequest - currpos))
    else:
        movedirection = 1
    return movedirection


# check for out of bounds request.  Cannot be < 0 or > 180
def checkboundary(torequest):
    if torequest < 0:
        torequest = 0
    elif torequest > 180:
        torequest = 180
    return torequest


# Init Pan Tilt Module and Center Camera
def init_pantilt():
    pwm = PCA9685()
    pwm.setPWMFreq(50)
    currpan, currtilt = center_pantilt(pwm)
    return currpan, currtilt, pwm


# set pan tilt to initial values
def center_pantilt(pwm):
    pancenter = 40
    tiltcenter = 70
    pwm.setRotationAngle(1, tiltcenter)
    pwm.setRotationAngle(0, pancenter)
    return pancenter, tiltcenter


if __name__ == "__main__":
    pancenter, tiltcenter, pwm = init_pantilt()  # init device
    currpan, currtilt = setservoangle(pwm, pancenter, 80, tiltcenter, 120)  # normal move
    currpan, currtilt = setservoangle(pwm, currpan, 20, currtilt, 20)  # normal move
    currpan, currtilt = setservoangle(pwm, currpan, -360, currtilt, 360)  # boundary check
    currpan, currtilt = setservoangle(pwm, currpan, currpan, currtilt, currtilt)  # divide zero check
    currpan, currtilt = setservoangle(pwm, currpan, pancenter, currtilt, tiltcenter)  # return to init state
