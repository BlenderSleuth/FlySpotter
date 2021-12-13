import os
from pathlib import Path
import cv2 as cv
from find_fly import find_fly, debug_draw_circles

## Options
fly_stationary_threshold = 20

def input_valid_directory_path() -> Path:
    valid_directory = False
    path = None
    while not valid_directory:
        top_image_directory = input("Path to images: ")
        try:
            path = Path(top_image_directory)
            if path.exists() and path.is_dir():
                valid_directory = True
            else:
                raise TypeError
        except TypeError:
            print("Please input a valid directory path.")

    return path

def main():
    top_image_path = input_valid_directory_path()

    fly_positions = dict()

    # Find all subdirectories
    plate_cell_paths = [x for x in top_image_path.iterdir() if x.is_dir()]
    for plate_cell_path in plate_cell_paths:
        plate_cell_id = str(plate_cell_path.name)

        # New dictionary entry
        fly_positions[plate_cell_id] = []

        # Find all images in this subdirectory
        image_paths = [x for x in plate_cell_path.iterdir() if x.is_file() and x.suffix == ".jpg"]

        for image_path in image_paths:
            fly_positions[plate_cell_id].append(str(image_path))

    print(fly_positions)

    # images = []
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     print(filename)
    #     if file
    #     if filename.endswith(".jpg"):
    #         images.append(os.fsdecode(os.path.join(directory, file)))
    #
    # fly_positions = []
    # for image_path in images:
    #     fly_pos = find_fly(image_path)
    #     fly_positions.append(fly_pos)
    #     print(f"Fly position: {fly_pos}")
    #
    # fly_deltas = []



def check_plate_circles():
    top_image_path = Path("C:/Users/Ben/Desktop/robottrial")
    image_dir_name = top_image_path.name
    dest_dir = "robottrial_circle_debug"
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
    check_plate_circles()