import os
from pathlib import Path
import numpy as np
from math import sqrt
from parse import parse
import cv2 as cv
from find_fly import find_fly
from csv import writer
from datetime import datetime
from typing import List, Tuple, Optional
from tqdm import tqdm

# FORMAT STRINGS

# Format string to extract data from plate-well directory
dir_format_str = "{plate_y:1d}-{plate_x:1d}-{well_alpha:l}{well_num:1d}"
# Format string to extract data from image filename
image_format_str = "{img_number:d}-{plate_y:1d}-{plate_x:1d}-{well_alpha:l}{well_num:1d}-{year:4d}_{month:2d}_{day:2d}_{hour:2d}_{minute:2d}_{second:2d}-0"
# Output CSV date format
date_format_str = "%Y-%m-%d %H:%M:%S"


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


def find_death_date(plate_well_path: Path,
                    stationary_threshold=10,
                    death_day_threshold=5) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Finds the start and death date of a fly in a well
    :param plate_well_path: path to the images for this well
    :param stationary_threshold: minimum distance a fly must move between hours to be considered stationary
    :param death_day_threshold: minimum number of days a fly must be stationary to be considered dead
    """

    # Find all images of the fly in this well
    image_paths: List[Path] = [x for x in plate_well_path.iterdir() if x.is_file()]

    if len(image_paths) == 0:
        print(f"No images found for {plate_well_path.stem}")
        return None, None

    is_blank_well = True

    # Collect fly positions in this well
    fly_positions: List[Tuple[datetime, Tuple[int, int]]] = []
    for image_path in tqdm(image_paths, desc=f" {plate_well_path.stem}"):
        # Extract date and time from file name
        image_details = parse(image_format_str, image_path.stem)

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

        if fly_pos:
            # Found a fly, not a blank well
            is_blank_well = False

        # TODO: Manual select
        # else:
        #     # If failed, assume last position
        #     if len(fly_positions) > 0:
        #         fly_pos = fly_positions[-1][1]
        #     else:
        #         fly_pos = (0, 0) # Start at (0, 0)

        # Append new position
        fly_positions.append((image_datetime, fly_pos))

    if is_blank_well or len(fly_positions) == 0:
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
        start_from: Optional[str]=None):
    """

    :param top_image_path: path to the directory containing all the output images
    :param output_csv_filepath: path to the output csv file
    :param save_continuously: write to output CSV file after each well is processed, rather than at the end
    :param start_from: plate-well to start from (to resume processing after crash)
    """

    # Data to output to CSV
    #                   plate, well, start, death, num hours
    # output_rows: List[Tuple[int, str, str, str, str]] = []

    with open(str(output_csv_filepath), 'w', newline='') as output_file:
        csv_writer = writer(output_file)

        # Write header
        csv_writer.writerow(["Plate Number", "Well", "Start Date", "Date of Death", "Num Hours"])

        # Find all subdirectories
        plate_well_paths = [x for x in top_image_path.iterdir() if x.is_dir()]

        # Sort in ascending order by plate-well
        plate_well_paths.sort(key=lambda x: x.stem)

        # Find directory to start from
        if start_from is not None:
            start_idx = 0
            for i, plate_well_path in enumerate(plate_well_paths):
                if plate_well_path.stem == start_from:
                    start_idx = i
                    break

            # Truncate start of list
            plate_well_paths = plate_well_paths[start_idx:]

        for plate_well_path in tqdm(plate_well_paths, desc="Total: "):

            # Extract plate and well from folder name
            dir_details = parse(dir_format_str, plate_well_path.stem)
            plate_num = plate_coord_to_num(dir_details["plate_x"], dir_details["plate_y"])
            well_coord = dir_details["well_alpha"] + str(dir_details["well_num"])

            # Find the start and death date of the fly in the well
            start_date, death_date = find_death_date(plate_well_path)

            if start_date is None:
                # Empty folder or blank well
                continue

            start_date_str = start_date.strftime(date_format_str)

            if death_date is None:
                death_date_str = "N/A"
                num_hours_str = "N/A"
            else:
                death_date_str = death_date.strftime(date_format_str)
                time_delta = death_date - start_date
                num_hours_str: str = str(round(time_delta.total_seconds() / 360))

            output_row = (plate_num, well_coord, start_date_str, death_date_str, num_hours_str)
            # output_rows.append(output_row)

            # Write to output CSV
            csv_writer.writerow(output_row)

            if save_continuously:
                output_file.flush()

def main(output_filename: str = "flyspotter_output.csv"):
    # Get input and output directories
    top_image_path = input_valid_directory_path("Path to images: ")
    output_csv_file_path = Path(input_valid_directory_path("Path to CSV output folder: ")) / output_filename

    run(top_image_path, output_csv_file_path, save_continuously=True)




def check_plate_circles():
    top_image_path = Path("C:/Users/Ben/Desktop/robottrial")
    image_dir_name = top_image_path.name
    dest_dir = "robottrial_circle_debug_2"
    #dest_path = top_image_path.with_name(dest_dir)

    plate_cell_paths = [x for x in top_image_path.iterdir() if x.is_dir()]

    for plate_cell_path in plate_cell_paths:
        plate_cell_id = str(plate_cell_path.name)

        # Find all images in this subdirectory
        image_paths = [x for x in plate_cell_path.iterdir() if x.is_file() and x.suffix == ".jpg"]

        for image_path in image_paths:
            image_dest_path_str = str(image_path).replace(image_dir_name, dest_dir)
            image_dest_path = Path(image_dest_path_str)
            image_dest_path.parent.mkdir(parents=True, exist_ok=True)

            circle_image = find_fly(str(image_path), crop_margin=-10, draw_debug_result=True, return_debug_result=True)
            if circle_image is not None:
                cv.imwrite(image_dest_path_str, circle_image)



if __name__ == "__main__":
    start_time = datetime.now()
    main()
    delta_time = datetime.now() - start_time
    print(f"Time elapsed: {delta_time} seconds")