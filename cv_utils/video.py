import cv2 as cv
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
            four_cc_int=self.capture_object.get(cv.CAP_PROP_FOURCC)
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

class VideoOutput:
    def __init__(
        self,
        output_path,
        video_parameters
    ):
        self.video_parameters = video_parameters
        self.writer_object = cv.VideoWriter(
            output_path,
            fourcc=self.video_parameters.four_cc_int,
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
        four_cc_int=None,
        time_index=None
    ):
        self.start_time=None
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.frame_count = None
        self.four_cc_int = None
        self.time_index = None
        if start_time is not None:
            try:
                self.start_time = start_time.astimezone(datetime.timezone.utc)
            except:
                try:
                    self.start_time = datetime.fromisoformat(start_time).astimezone(datetime.timezone.utc)
                except:
                    raise ValueError('Cannot parse start time: {}'.format(start_time))
        if frame_width is not None:
            try:
                self.frame_width = int(frame_width)
            except:
                raise ValueError('Frame width must be convertible to integer')
        if frame_height is not None:
            try:
                self.frame_height = int(frame_height)
            except:
                raise ValueError('Frame height must be convertible to integer')
        if fps is not None:
            try:
                self.fps = float(fps)
            except:
                raise ValueError('FPS must be convertible to float')
        if frame_count is not None:
            try:
                self.frame_count = int(frame_count)
            except:
                raise ValueError('Frame count must be convertible to integer')
        if four_cc_int is not None:
            try:
                self.four_cc_int = int(four_cc_int)
            except:
                raise ValueError('FourCC code must be convertible to integer')
        if time_index is not None:
            try:
                self.time_index = [timestamp.astimezone(datetime.timezone.utc) for timestamp in time_index]
            except:
                try:
                    self.time_index = [datetime.fromisoformat(timestamp).astimezone(datetime.timezone.utc) for timestamp in time_index]
                except:
                    raise ValueError('Cannot parse time index')
        if self.time_index is None and self.start_time is not None and self.fps is not None and self.frame_count is not None:
            self.time_index = [self.start_time + index*datetime.timedelta(microseconds = int(10**6/self.fps)) for index in range(self.frame_count)]
