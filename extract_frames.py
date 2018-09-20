import os
import random

import av
from PIL import Image

base_dir_path = os.path.dirname(os.path.realpath(__file__))

video_ids = sorted([5833084561001,
                    5831137888001,
                    5833096735001,
                    5833090951001,
                    5831135399001,
                    5831139883001,
                    5831138807001,
                    ])

images_per_video = 200
image_height_and_width = 224  # Most transfer networks we woudl want to use have this as input size

with open("{}/data-labels/images.csv".format(base_dir_path), "w") as metadata_file:
    metadata_file.write("\"VIDEO ID\",\"IMAGE ID\",\"IMAGE FILENAME\",\"IS HUDDLE\"\n")

    for video_id in video_ids:
        video = av.open("{}/data-videos/{}.mp4".format(base_dir_path, video_id))
        image_samples = images_per_video

        print("Video {}: duration: {} seconds".format(video_id, int(video.duration / av.time_base)))

        frame_positions = sorted(random.sample(range(int(video.duration / av.time_base)), k=image_samples))

        for frame_position in frame_positions:
            image_id = frame_position
            image_file_name = "{}-{}.jpg".format(video_id, image_id)
            image_file_path = "{}/data-images/{}".format(base_dir_path, image_file_name)

            video.seek(frame_position * av.time_base)
            frame = next(video.decode(0))
            image = frame.to_image()

            height_ratio = float(image_height_and_width / float(image.size[1]))
            tmp_image_width = int((float(image.size[0]) * height_ratio))
            size = (tmp_image_width, image_height_and_width)
            image = image.resize(size, Image.ANTIALIAS)

            left_margin = int((tmp_image_width - image_height_and_width) / 2)
            image_crop_box = (
                left_margin, 0,
                left_margin + image_height_and_width, image_height_and_width
            )

            image = image.crop(image_crop_box)
            image.save(image_file_path, format='JPEG')

            metadata_file.write("\"{}\",\"{}\",\"{}\",\n".format(video_id, frame_position, image_file_name, ))
