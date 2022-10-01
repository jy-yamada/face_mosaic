import cv2
import os
import sys
import copy
import numpy as np
from PIL import Image

valid_extensions = ["jpeg", "jpg", "png"]

def path_validator(img_path):
    if img_path == "exit":
        sys.exit()
    if not os.path.isfile(img_path):
        print ("Incorrect file path. Only accepts an absolute file path")
        return False
    if img_path.split(".")[-1].lower() not in valid_extensions:
        valid_extensions_string = (", ").join(valid_extensions)
        print (f"Invalid extension. Extensions we support are as follows: {valid_extensions_string}")
        return False
    return True

def arguments_validator(arguments_array):
    if arguments_array[0] == "exit":
        sys.exit()
    if len(arguments_array) < 2 or len(arguments_array) > 2:
        print ("Invalid input. Exactly 2 values required")
        return False
    for ele in arguments_array:
        if not ele.isdigit() or int(ele) < 1:
            print ("Enter positive integers")
            return False
    return True

def convertToMosaic(img, x, y, x_mosaic_range, y_mosaic_range):
    target_pixels = img[y : y+y_mosaic_range, x : x+x_mosaic_range]
    blue_sum = 0
    green_sum = 0
    red_sum = 0
    height = img.shape[0]
    width = img.shape[1]

    for vicinity_y in range(y, y + y_mosaic_range):
        if vicinity_y >= height: continue
        for vicinity_x in range(x, x + x_mosaic_range):
            if vicinity_x >= width: continue
            current_vicinity_pixel = img[vicinity_y][vicinity_x]
            blue_sum += current_vicinity_pixel[0]
            green_sum += current_vicinity_pixel[1]
            red_sum += current_vicinity_pixel[2]

    number_of_pixels = y_mosaic_range * x_mosaic_range
    average_blue = int(blue_sum / number_of_pixels)
    average_green = int(green_sum / number_of_pixels)
    average_red = int(red_sum / number_of_pixels)

    target_pixels = [average_blue, average_green, average_red]

    return target_pixels

print ("─────────────────────")
print ("//// FACE MOSAIC ////")
print ("─────────────────────")
print ("*To exit the program, enter \"exit\"\n")

while True:
    img_path = None
    mosaic_ranges = None

    while img_path == None or not path_validator(img_path):
        img_path = input("Absolute image file path you want to edit: ").strip()

    while mosaic_ranges == None or not arguments_validator(mosaic_ranges):
        mosaic_ranges = input ("X-block-width(pixel), y-block-width(pixel)(ex.:\"10 10\"): ").strip().split(" ")

    pil_img = Image.open(img_path)
    img_ndarray = np.array(pil_img)
    mono = img_ndarray
    img_source = None
    if img_ndarray.ndim == 3:
        img_source = cv2.cvtColor(img_ndarray, cv2.COLOR_RGB2BGR)
        mono = cv2.cvtColor(img_ndarray, cv2.COLOR_RGB2GRAY)
    else:
        img_source = cv2.cvtColor(img_ndarray, cv2.COLOR_GRAY2BGR)
    img_copy = copy.deepcopy(img_source)

    path = "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(path)
    targets = classifier.detectMultiScale(mono)

    x_mosaic_range = int(mosaic_ranges[0])
    y_mosaic_range = int(mosaic_ranges[1])

    for x, y, w, h in targets:
        for column in range(y, y+h, y_mosaic_range):
            for row in range(x, x+w, x_mosaic_range):
                img_copy[column:column+y_mosaic_range, row:row+x_mosaic_range] = convertToMosaic(img_source, row, column, x_mosaic_range, y_mosaic_range)

    img_name = img_path.split(".")[0:-1]
    img_name = ("").join(img_name)

    print ("processing...")
    result_pil_image = Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    save_path = f"{img_name}_mosaic.png"

    try:
        i = 0
        while os.path.isfile(save_path):
            i += 1
            save_path = f"{img_name}_mosaic({i}).png"
        status = result_pil_image.save(save_path, quality=95)
        print ("Created the image successfully")
        print (f"File path: {save_path}")

    except:
        print ("Failed to create the image")
    
    print ("─────────────────────────")