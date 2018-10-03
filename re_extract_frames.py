import os
import random

import av
from PIL import Image

base_dir_path = os.path.dirname(os.path.realpath(__file__))

# Height of all videos is currently 270 and we will use a 270x270 crop.
image_height_and_width = 270

with open("{}/data-labels/images.csv".format(base_dir_path), "r") as metadata_file:

    videp = None
    last_video_id = None
    for line in metadata_file.readlines():
        if 'VIDEO ID' in line:
            continue # skip over the header

        line = line.replace('"', '')
        line_parts = line.split(",")

        video_id = int(line_parts[0])
        image_id = int(line_parts[1])

        if video_id != last_video_id:
            video = av.open("{}/data-videos/{}.mp4".format(base_dir_path, video_id))
            last_video_id = video_id

        image_file_name = "{}-{}.jpg".format(video_id, image_id)
        image_file_path = "{}/data-images/{}".format(base_dir_path, image_file_name)

        frame_position = image_id
        video.seek(frame_position * av.time_base)
        frame = next(video.decode(0))
        image = frame.to_image()

        current_width, current_height = image.size
        new_width = current_height
        new_height = current_height

        left_margin = int((current_width - new_width) / 2)
        image_crop_box = (
            left_margin, 0,
            left_margin + new_width, new_height
        )

        image = image.crop(image_crop_box)
        image.save(image_file_path, format='JPEG')
