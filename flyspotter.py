"""
FlySpotter: Script to automate Drosophila starvation assays

Takes the top image directory as input and outputs a CSV file with the date of death for each fly in a well

@see run() and main()
"""

from pathlib import Path
from math import sqrt
from multiprocessing import Pool
from datetime import datetime
from csv import writer
from typing import List, Tuple, Optional
import numpy as np
import cv2 as cv
from parse import compile
from tqdm import tqdm

# CONSTANTS
SQRT_2 = sqrt(2.)

# FORMAT STRINGS

# Format string to extract data from plate-well directory
dir_format_str = \
    compile("{plate_y:1d}-{plate_x:1d}-{well_alpha:l}{well_num:1d}")

# Format string to extract data from image filename
image_format_str = \
    compile("{img_number:d}-{plate_y:1d}-{plate_x:1d}-{well_alpha:l}{well_num:1d}-{year:4d}_{month:2d}_{day:2d}_{hour:2d}_{minute:2d}_{second:2d}-{:d}")

# Output CSV date format
date_format_str = "%Y-%m-%d %H:%M:%S"


def find_fly(image_path: str,
             crop_margin=-10,
             min_circle_radius=200,
             max_circle_radius=400,
             fly_radius_threshold=10,
             circle_radius_override=250,
             draw_debug_crop=False,
             draw_debug_threshold=False,
             draw_debug_contours=False,
             draw_debug_result=False,
             return_debug_result=False) -> Optional[Tuple[int, int]]:
    """
    Finds the position of a fly in an image, by first cropping to a standard size. Defaults tested on 800x600 image size
    :param image_path: path to the image
    :param crop_margin: margin around plate circle to use for cropping to standard size (in pixels)
    :param min_circle_radius: minimum potential size of plate circle (in pixels)
    :param max_circle_radius: maximum potential size of plate circle (in pixels)
    :param circle_radius_override: use constant circle radius size (recommended)
    :param fly_radius_threshold: smallest allowable radius of fly in pixels
    :param draw_debug_crop: show cropping debug detail
    :param draw_debug_threshold: show threshold operation debug detail
    :param draw_debug_contours: show contour operation debug detail
    :param draw_debug_result: show position of fly
    :param return_debug_result: return the debug image showing the calculate position of the fly
    :return: tuple of 2 ints: position of the fly in the cropped image (standardised), or None on failure
    """

    input_img = cv.imread(image_path)

    if input_img is None:
        print(f"\nError: Image is corrupted, skipping: {image_path}")
        return

    # Make greyscale
    greyscale_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

    # Increase contrast phase (helps with circle detection)
    greyscale_img = cv.convertScaleAbs(greyscale_img, alpha=2, beta=-50)

    # Get image dimensions
    img_height, img_width = greyscale_img.shape

    # Detect circles in the image
    #                                                 
    circles = cv.HoughCircles(greyscale_img,        # image
                              cv.HOUGH_GRADIENT,    # method
                              1.2,                  # accumulator
                              100,                  # minDist
                              minRadius=min_circle_radius, 
                              maxRadius=max_circle_radius)

    if circles is None or len(circles) == 0:
        print(f"\nError: Couldn't find any circles in image: {image_path}")
        return

    if len(circles) > 1:
        print(f"\nError: Too many circles in image: {image_path}")
        return

    # convert the (x, y) coordinates and radius of the circles to integers, and find crop rectangle
    circle_x, circle_y, circle_radius = np.round(circles[0, :]).astype("int")[0]

    # Make constant radius between shapes
    if circle_radius_override is not None:
        circle_radius = circle_radius_override

    crop_radius = circle_radius + crop_margin
    crop_rect = [circle_x - crop_radius, circle_y - crop_radius, circle_x + crop_radius, circle_y + crop_radius]
    
    crop_centre = (crop_radius, crop_radius)  # Centre in cropped coordinates

    if draw_debug_crop:
        # draw debug circle and crop rectangle
        debug_img = input_img.copy()
        cv.circle(debug_img, (circle_x, circle_y), circle_radius, (0, 255, 0), 4)
        cv.rectangle(debug_img, (crop_rect[0], crop_rect[1]),(crop_rect[2], crop_rect[3]), (0, 128, 255), thickness=5)

        if not return_debug_result:
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
    padded = cv.copyMakeBorder(input_img, *padding, cv.BORDER_CONSTANT, value=(255, 255, 255))

    # To crop image in rect:
    # (min_x, max_x, min_y, max_y): image[min_y:max_y, min_x:max_x]
    cropped = padded[crop_rect[1]:crop_rect[3],crop_rect[0]:crop_rect[2]].copy()

    # White out circles
    radius = int(crop_radius*SQRT_2)
    circle_margin = 10
    thickness = int(2*crop_radius*(SQRT_2-1))+circle_margin
    cv.circle(cropped, crop_centre, radius, (255, 255, 255), thickness=thickness)

    # Find binary threshold
    grey_cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    blurred_cropped = cv.GaussianBlur(grey_cropped, (11, 11), sigmaX=5, borderType=cv.BORDER_DEFAULT)
    _, thresh = cv.threshold(blurred_cropped, 100, 255, type=cv.THRESH_BINARY_INV)

    if draw_debug_threshold and not return_debug_result:
        cv.imshow("Threshold", thresh)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # print(f"\nError: no contours in image {image_path}")
        return

    # Find largest contour by area (which is the fly)
    largest_contour = max(contours, key=cv.contourArea)

    ((fly_x, fly_y), fly_radius) = cv.minEnclosingCircle(largest_contour)

    if fly_radius <= fly_radius_threshold:
        # print(f"\nError: fly too small in image {image_path}")
        return

    fly_centre = (int(fly_x), int(fly_y))
    fly_radius = int(fly_radius)

    if draw_debug_contours:
        cropped = cv.drawContours(cropped, contours, -1, (0, 255, 0), 3)
        if not return_debug_result:
            cv.imshow("Debug Contours", cropped)

    # Draw where the fly is
    if draw_debug_result:
        cv.circle(cropped, fly_centre, fly_radius, (255, 0, 100), 4)
        cv.rectangle(cropped, (fly_centre[0]-5, fly_centre[1]-5), (fly_centre[0] + 5, fly_centre[1]+5), color=(0, 0, 255), thickness=cv.FILLED)

    if return_debug_result:
        return cropped
    elif draw_debug_result:
        # Show the output image
        cv.imshow("Output", cropped)

    if draw_debug_crop or draw_debug_threshold or draw_debug_contours or draw_debug_result:
        cv.waitKey(0)

    return fly_centre


def plate_coord_to_num(plate_x: int, plate_y: int) -> int:
    """
    Converts a plate grid coordinate (1-1 etc.) to a number (9). Assumes a 3x3 grid

    +-----+-----+-----+     +---+---+---+
    | 1-1 | 1-2 | 1-3 |     | 1 | 2 | 3 |
    +-----+-----+-----+     +---+---+---+
    | 2-1 | 2-2 | 2-3 | ->  | 4 | 5 | 6 |
    +-----+-----+-----+     +---+---+---+
    | 3-1 | 3-2 | 3-3 |     | 7 | 8 | 9 |
    +-----+-----+-----+     +---+---+---+

    :param plate_x: x-coordinate
    :param plate_y: y-coordinate
    :return: corresponding plate number
    """
    return plate_x + 3*(plate_y-1)


def input_valid_directory_path(prompt: str) -> Path:
    """
    Input loop to validate a desired directory
    :param prompt: text to display to user
    :return: a valid Path object of a directory on disk
    """
    valid_directory = False
    path = None
    while not valid_directory:
        top_image_directory = input(prompt)
        try:
            path = Path(top_image_directory)
            if path.exists() and path.is_dir():
                valid_directory = True
            else:
                raise TypeError
        except TypeError:
            print("Please input a valid directory path.")

    return path


def dist(last_pos: Tuple[int, int], pos: Tuple[int, int]) -> int:
    """
    Finds the distance between two pixel positions
    :param last_pos: position last frame
    :param pos: position this frame
    :return: distance between positions, rounded
    """
    dx2: int = (last_pos[0] - pos[0]) * (last_pos[0] - pos[0])
    dy2: int = (last_pos[1] - pos[1]) * (last_pos[1] - pos[1])
    return round(sqrt(dx2 + dy2))


def process_image(image_path: Path) -> Tuple[datetime, Tuple[int, int]]:
    """
    Processes a single image of a fly in a well
    :param image_path: Path to the image
    :return: A tuple with the date the image was taken and the position of the fly in the image
    """

    # Extract date and time from file name
    image_details = image_format_str.parse(image_path.stem)

    # Create datetime object from file name
    image_datetime = datetime(image_details["year"],
                              image_details["month"],
                              image_details["day"],
                              image_details["hour"],
                              image_details["minute"],
                              image_details["second"])

    # Find the fly in this image
    try:
        fly_pos = find_fly(str(image_path))
    except Exception as e:
        print(f"Exception occurred while processing image {image_path}")
        raise e

    # TODO: Manual select
    # else:
    #     # If failed, assume last position
    #     if len(fly_positions) > 0:
    #         fly_pos = fly_positions[-1][1]
    #     else:
    #         fly_pos = (0, 0) # Start at (0, 0)

    # Return new position
    return image_datetime, fly_pos


def find_death_date(plate_well_path: Path,
                    stationary_threshold=10,
                    death_day_threshold=5,
                    multithreaded=True) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Finds the start and death date of a fly in a well
    :param plate_well_path: path to the images for this well
    :param stationary_threshold: minimum distance a fly must move between hours to be considered stationary
    :param death_day_threshold: minimum number of days a fly must be stationary to be considered dead
    :param multithreaded: run on multiple threads (faster, progress bars less predictable)
    """

    # Find all images of the fly in this well
    image_paths: List[Path] = [x for x in plate_well_path.iterdir() if x.is_file()]

    if len(image_paths) == 0:
        print(f"No images found for {plate_well_path.stem}")
        return None, None

    # Collect fly positions in this well

    # Run process_image() on every image, using multithreading
    if multithreaded:
        with Pool() as p:
            fly_positions = list(tqdm(p.imap_unordered(process_image, image_paths),
                                      total=len(image_paths),
                                      desc=f" {plate_well_path.stem}"))
    else:
        fly_positions = list(tqdm(map(process_image, image_paths),
                                  total=len(image_paths),
                                  desc=f" {plate_well_path.stem}"))

    if len(fly_positions) == 0:
        # Empty folder, return nothing
        return None, None

    # Check for blank well (all positions are None)
    for _, pos in fly_positions:
        if pos is not None:
            break
    # 'else' block runs when loop does not break
    else:
        # Blank well, return nothing
        return None, None

    # Sort by date and time, in ascending order
    fly_positions.sort()

    start_date: datetime = fly_positions[0][0]

    # Store the distance between fly positions between frames
    fly_deltas: List[Tuple[datetime, int]] = []
    last_pos = fly_positions[0][1]
    for image_time, fly_pos in fly_positions[1:]:
        if fly_pos is None:
            # Ignore blank fly positions
            continue
        if last_pos is None:
            last_pos = fly_pos
            continue

        fly_deltas.append((image_time, dist(last_pos, fly_pos)))
        last_pos = fly_pos

    # Find date of death based on thresholds
    streak: int = 0
    death_date = start_date
    for image_time, fly_delta in fly_deltas:
        if not fly_delta:
            # Ignore failed images
            continue

        if fly_delta < stationary_threshold:
            streak += 1
            if streak >= death_day_threshold:
                return start_date, death_date
        else:
            death_date = image_time
            streak = 0

    return start_date, None


def run(top_image_path: Path = Path.cwd,
        output_csv_filepath: Path = Path.cwd,
        save_continuously=True,
        start_from: Optional[str] = None):
    """
    Runs the script on the image directory.

    :param top_image_path: path to the directory containing all the output images
    :param output_csv_filepath: path to the output csv file
    :param save_continuously: write to output CSV file after each well is processed, rather than at the end
    :param start_from: plate-well to start from (to resume processing after crash)
    """

    # Data to output to CSV
    #                   plate, well, start, death, num hours
    # output_rows: List[Tuple[int, str, str, str, str]] = []

    with open(str(output_csv_filepath), 'a', newline='') as output_file:
        csv_writer = writer(output_file)

        # Find all subdirectories
        plate_well_paths = [x for x in top_image_path.iterdir() if x.is_dir()]

        # Sort in ascending order by plate-well
        plate_well_paths.sort(key=lambda x: x.stem)

        # Find directory to start from
        start_idx = 0
        if start_from is not None:
            for i, plate_well_path in enumerate(plate_well_paths):
                if plate_well_path.stem == start_from:
                    start_idx = i
                    break

            # Truncate start of list
            plate_well_paths = plate_well_paths[start_idx:]

        # Write header if starting from beginning
        if start_idx == 0:
            csv_writer.writerow(
                ["Plate Number", "Well", "Start Date", "Date of Death", "Num Hours"])

        for plate_well_path in tqdm(plate_well_paths, desc="Total: "):

            # Extract plate and well from folder name
            dir_details = dir_format_str.parse(plate_well_path.stem)
            if dir_details is None:
                # Ignore other folders
                continue
            
            plate_num = plate_coord_to_num(dir_details["plate_x"], dir_details["plate_y"])
            well_coord = dir_details["well_alpha"] + str(dir_details["well_num"])

            # Find the start and death date of the fly in the well
            start_date, death_date = find_death_date(plate_well_path)

            if start_date is None:
                start_date_str = "N/A"
            else:
                start_date_str = start_date.strftime(date_format_str)

            if death_date is None:
                death_date_str = "N/A"
                num_hours_str = "N/A"
            else:
                death_date_str = death_date.strftime(date_format_str)
                time_delta = death_date - start_date
                num_hours_str = str(round(time_delta.total_seconds() / 3600.))

            output_row = (plate_num,
                          well_coord,
                          start_date_str,
                          death_date_str,
                          num_hours_str)

            # Write to output CSV
            csv_writer.writerow(output_row)

            if save_continuously:
                output_file.flush()


def main(output_filename: str = "flyspotter_output.csv"):
    # Get input and output directories
    top_image_path = input_valid_directory_path("Path to images: ")
    start_from = input("Plate-well to start from (Empty defaults to 1-1-A1):")
    output_csv_file_path = Path(input_valid_directory_path(
        "Path to CSV output folder: ")) / output_filename

    run(top_image_path,
        output_csv_file_path,
        save_continuously=True,
        start_from=start_from)


def check_plate_circles():
    """
    Debug script to output calculated fly positions for a dataset.
    """
    top_image_path = input_valid_directory_path("Path to images: ")
    dest_dir = input_valid_directory_path("Path to destination: ")

    plate_cell_paths = [x for x in top_image_path.iterdir() if x.is_dir()]

    for plate_cell_path in plate_cell_paths:

        # Find all images in this subdirectory
        image_paths = [x for x in plate_cell_path.iterdir() if x.is_file() and x.suffix == ".jpg"]

        for image_path in image_paths:
            image_dest_path = dest_dir / plate_cell_path.name / image_path.name
            image_dest_path.parent.mkdir(parents=True, exist_ok=True)

            circle_image = find_fly(str(image_path), crop_margin=-10, draw_debug_contours=True, draw_debug_result=False, return_debug_result=True)
            if circle_image is not None:
                cv.imwrite(str(image_dest_path), circle_image)


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    delta_time = datetime.now() - start_time
    print(f"Time elapsed: {delta_time}")
