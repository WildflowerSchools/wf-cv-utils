import cv2 as cv
import pandas as pd
import datetime
import os


class VideoInput:
    def __init__(
        self,
        input_path,
        start_time=None
    ):
        if not os.path.isfile(input_path):
            raise ValueError('No file at specified path: {}'.format(input_path))
        self.capture_object = cv.VideoCapture(input_path)
        self.video_parameters = VideoParameters(
            start_time=start_time,
            frame_width=self.capture_object.get(cv.CAP_PROP_FRAME_WIDTH),
            frame_height=self.capture_object.get(cv.CAP_PROP_FRAME_HEIGHT),
            fps=self.capture_object.get(cv.CAP_PROP_FPS),
            frame_count=self.capture_object.get(cv.CAP_PROP_FRAME_COUNT),
            fourcc_int=self.capture_object.get(cv.CAP_PROP_FOURCC)
        )

    def is_opened(self):
        return self.capture_object.isOpened()

    def close(self):
        self.capture_object.release()

    def get_frame(self):
        ret, frame = self.capture_object.read()
        if ret:
            return frame
        else:
            return None

    def get_frame_by_timestamp(self, timestamp):
        if self.video_parameters.start_time is None or self.video_parameters.fps is None or self.video_parameters.frame_count is None:
            raise ValueError('Valid video start time, FPS, and frame count required to get frame by timestamp')
        try:
            timestamp = pd.to_datetime(timestamp, utc=True).to_pydatetime()
        except Exception as e:
            raise ValueError('Cannot parse start time: {}'.format(timestamp))
        frame_number = round(
            (timestamp -self.video_parameters.start_time).total_seconds()*
            self.video_parameters.fps
        )
        if frame_number < 0 or frame_number > self.video_parameters.frame_count:
            raise ValueError('Specified datetime is outside the time range of the video')
        return self.get_frame_by_frame_number(frame_number)

    def get_frame_by_frame_number(self, frame_number):
        self.capture_object.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        return self.get_frame()

    def get_frame_by_milliseconds(self, milliseconds):
        self.capture_object.set(cv.CAP_PROP_POS_MSEC, milliseconds)
        return self.get_frame()


class VideoOutput:
    def __init__(
        self,
        output_path,
        video_parameters
    ):
        self.video_parameters = video_parameters
        self.writer_object = cv.VideoWriter(
            output_path,
            fourcc=self.video_parameters.fourcc_int,
            fps=self.video_parameters.fps,
            frameSize=(
                self.video_parameters.frame_width,
                self.video_parameters.frame_height
            )
        )

    def is_opened(self):
        return self.writer_object.isOpened()

    def close(self):
        self.writer_object.release()

    def write_frame(self, frame):
        self.writer_object.write(frame)


class VideoParameters:
    def __init__(
        self,
        start_time=None,
        frame_width=None,
        frame_height=None,
        fps=None,
        frame_count=None,
        fourcc_int=None
    ):
        self.start_time = None
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.frame_count = None
        self.fourcc_int = None
        self.time_index = None
        if start_time is not None:
            try:
                self.start_time = pd.to_datetime(start_time, utc=True).to_pydatetime()
            except Exception as e:
                raise ValueError('Cannot parse start time: {}'.format(start_time))
            # try:
            #     self.start_time = start_time.astimezone(datetime.timezone.utc)
            # except Exception as e:
            #     try:
            #         self.start_time = datetime.fromisoformat(start_time).astimezone(datetime.timezone.utc)
                # except Exception as e:
                #     raise ValueError('Cannot parse start time: {}'.format(start_time))
        if frame_width is not None:
            try:
                self.frame_width = int(frame_width)
            except Exception as e:
                raise ValueError('Frame width must be convertible to integer')
        if frame_height is not None:
            try:
                self.frame_height = int(frame_height)
            except Exception as e:
                raise ValueError('Frame height must be convertible to integer')
        if fps is not None:
            try:
                self.fps = float(fps)
            except Exception as e:
                raise ValueError('FPS must be convertible to float')
        if frame_count is not None:
            try:
                self.frame_count = int(frame_count)
            except Exception as e:
                raise ValueError('Frame count must be convertible to integer')
        if fourcc_int is not None:
            try:
                self.fourcc_int = int(fourcc_int)
            except Exception as e:
                raise ValueError('FourCC code must be convertible to integer')

def fourcc_string_to_int(fourcc_string):
    return cv.VideoWriter_fourcc(*fourcc_string)

def fourcc_int_to_string(fourcc_int):
    return "".join([chr((int(fourcc_int) >> 8 * i) & 0xFF) for i in range(4)])
