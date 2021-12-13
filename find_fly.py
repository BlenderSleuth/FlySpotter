import numpy as np
import cv2 as cv
from pathlib import Path

def debug_draw_circles(imageName: str, show_result=False):
    input_img = cv.imread(imageName)
    greyscale_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

    #contrast = cv.convertScaleAbs(greyscale_img, alpha=2, beta=-50)

    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])


    #sharp_img = cv.filter2D(src=greyscale_img, ddepth=-1, kernel=sharpen_kernel)
    #blurred_img = cv.GaussianBlur(sharp_img, (7, 7), sigmaX=1.5, borderType=cv.BORDER_DEFAULT)
    circles = cv.HoughCircles(greyscale_img, method=cv.HOUGH_GRADIENT, dp=1.2, minDist=100)

    # cv.imshow("Sharp Debug", blurred_img)

    # Convert to int
    circles = np.round(circles[0, :]).astype("int")

    for x,y,r in circles:
        #print(x, y, r)
        cv.circle(input_img, (x, y), 250, (0, 255, 0), 4)

    if show_result:
        cv.imshow("Circle Debug", input_img)
        cv.waitKey(0)

    return input_img




def find_fly(image_path: str,
             crop_margin = 0,
             min_circle_radius = 200,
             max_circle_radius = 400,
             circle_radius_override = 250,
             draw_debug_crop=False,
             draw_debug_threshold=False,
             draw_debug_contours=False,
             draw_debug_result=False,
             return_debug_result=False) -> (int, int):
    """
    Finds the position of a fly in an image, by first cropping to a standard size.
    :param image_path: path to the image
    :param crop_margin: margin around plate circle to use for cropping to standard size (in pixels)
    :param min_circle_radius: minimum potential size of plate circle (in pixels)
    :param max_circle_radius: maximum potential size of plate circle (in pixels)
    :param circle_radius_override: use constant circle radius size (recommended)
    :param draw_debug_crop: show cropping debug detail
    :param draw_debug_threshold: show threshold operation debug detail
    :param draw_debug_contours: show contour operation debug detail
    :param draw_debug_result: show position of fly
    :param return_debug_result: return the debug image of the fly
    :return: tuple of 2 ints: position of the fly in the cropped image (standardised), or None on failure
    """

    input_img = cv.imread(image_path)

    # Contrast phase (helps with circle detection)
    #contrast = cv.convertScaleAbs(input_img, alpha=2, beta=-50)

    # Sharpen phase (also helps with circle detection)
    # https://en.wikipedia.org/wiki/Kernel_(image_processing)
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

    #sharp_img = cv.filter2D(src=input_img, ddepth=-1, kernel=sharpen_kernel)
    #cv.imshow("Sharp Debug", sharp_img)

    # Make greyscale
    greyscale_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

    # Get image dimensions
    img_height, img_width = greyscale_img.shape

    # Detect circles in the image
    #                         image             method       accumulator    minDist
    circles = cv.HoughCircles(greyscale_img, cv.HOUGH_GRADIENT,  1.2,         100, minRadius=min_circle_radius, maxRadius=max_circle_radius)

    if circles is None or len(circles) == 0:
        print(f"Couldn't find any circles in image: {image_path}")
        return

    if len(circles) > 1:
        print(f"Too many circles in image: {image_path}")
        return

    # convert the (x, y) coordinates and radius of the circles to integers, and find crop rectangle
    circle_x, circle_y, circle_radius = np.round(circles[0, :]).astype("int")[0]

    if circle_radius_override is not None:
        circle_radius = circle_radius_override

    crop_radius = circle_radius + crop_margin
    crop_rect = [circle_x - crop_radius, circle_y - crop_radius, circle_x + crop_radius, circle_y + crop_radius]

    if draw_debug_crop:
        # draw debug circle and crop rectangle
        debug_img = input_img.copy()
        cv.circle(debug_img, (circle_x, circle_y), circle_radius, (0, 255, 0), 4)
        cv.rectangle(debug_img, (crop_rect[0], crop_rect[1]),(crop_rect[2], crop_rect[3]), (0, 128, 255), thickness=5)
        cv.imshow("Crop Debug", debug_img)

    # Pad image with white to allow cropping to a square
    padding = list(map(lambda a: max(a, 0), [
        -crop_rect[1],               # Top
        crop_rect[3] - img_height,   # Bottom
        -crop_rect[0],               # Left
        crop_rect[2] - img_width     # Right
    ]))

    # Shift y by top padding
    crop_rect[1] += padding[0]
    crop_rect[3] += padding[0]
    # Shift x by left padding
    crop_rect[0] += padding[2]
    crop_rect[2] += padding[2]
    padded = cv.copyMakeBorder(input_img, *padding, cv.BORDER_CONSTANT, value=[255,255,255])

    # To crop image in rect (min_x, max_x, min_y, max_y): image[min_y:max_y, min_x:max_x]
    cropped = padded[crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]].copy()

    # Find binary threshold
    grey_cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    blurred_cropped = cv.GaussianBlur(grey_cropped, (11, 11), sigmaX=5, borderType=cv.BORDER_DEFAULT)
    _, thresh = cv.threshold(blurred_cropped, 100, 255, type=cv.THRESH_BINARY_INV)

    if draw_debug_threshold:
        cv.imshow("Threshold", thresh)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)#, offset=(x-r-crop_margin,y-r-crop_margin))

    # Find largest contour by area (which is the fly)
    largest_contour = max(contours, key=cv.contourArea)

    ((fly_x, fly_y), fly_radius) = cv.minEnclosingCircle(largest_contour)

    if fly_radius <= 2:
        print("Error: fly too small")
        return

    fly_centre = (int(fly_x), int(fly_y))
    fly_radius = int(fly_radius)

    if draw_debug_contours:
        cont_img = cv.drawContours(cropped.copy(), contours, -1, (0, 255, 0), 3)
        cv.imshow("Debug Contours", cont_img)

    # Draw where the fly is
    if draw_debug_result:
        cv.circle(cropped, fly_centre, fly_radius, (255, 0, 100), 4)
        cv.rectangle(cropped, (fly_centre[0]-5,fly_centre[1]-5), (fly_centre[0]+5,fly_centre[1]+5), color=(0, 0, 255), thickness=cv.FILLED)

    if return_debug_result:
        return cropped
    elif draw_debug_result:
        # Show the output image
        cv.imshow("Output", cropped)

    if draw_debug_crop or draw_debug_threshold or draw_debug_contours or draw_debug_result:
        cv.waitKey(0)

    return fly_centre

if __name__ == "__main__":
    image_path = "C:/Users/Ben/Desktop/robottrial/1-1-E1/22-1-1-E1-2021_11_27_07_41_35-0.jpg"

    # Requires initial image processing to detect circles:
    # "C:/Users/Ben/Desktop/robottrial/1-1-F5/7-1-1-F5-2021_11_26_16_41_57-0.jpg"

    #debug_draw_circles(image_path, show_result=True)

    fly_pos = find_fly(
        image_path,
        crop_margin=-10,
        min_circle_radius=0,
        max_circle_radius=400000,
        draw_debug_crop=True,
        draw_debug_threshold=True,
        draw_debug_contours=True,
        draw_debug_result=True
    )

    # print(f"Fly position: {fly_pos}")

    # fly_1: (539, 446)     (539, 444)
    # fly_2: (547, 452)     (547, 450)
    # fly_3: (540, 433)     (540, 431)
    # fly_4: (556, 449)     (556, 447)

