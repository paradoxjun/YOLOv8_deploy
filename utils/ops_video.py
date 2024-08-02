import cv2
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

import values.error_type as error_common
from values.strings import legal_url_v1


def get_cap(video_input):
    read_type = "unknown"

    if isinstance(video_input, str):
        try:
            if video_input.startswith("rtsp"):
                read_type = "RTSP"
            elif legal_url_v1.match(video_input):
                read_type = "HTTP"
            else:
                read_type = "PATH"

            cap = cv2.VideoCapture(video_input)
            if not cap.isOpened():
                raise error_common.ReadVideoError

        except Exception:
            raise error_common.ReadVideoError(f"{read_type}: Get cap failed, unable to open video stream. ")
    else:
        raise error_common.InputFormatError(f"Error: Get invalid input format {type(video_input)}")

    return cap
