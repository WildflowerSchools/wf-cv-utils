import cv_utils.core
import cv_utils.eager_video_capture as vc_queue
import cv2
import pandas as pd
import datetime
import os
import logging

logger = logging.getLogger(__name__)


class VideoInput:
    def __init__(
            self,
            input_path,
            start_time=None,
            queue_frames=True,
            queue_size=64,
            use_cvcuda=False,
            use_gpu=False
    ):
        if not os.path.isfile(input_path):
            raise ValueError('No file at specified path: {}'.format(input_path))

        video_details = cv2.VideoCapture(input_path)
        self.video_parameters = VideoParameters(
            start_time=start_time,
            frame_width=video_details.get(cv2.CAP_PROP_FRAME_WIDTH),
            frame_height=video_details.get(cv2.CAP_PROP_FRAME_HEIGHT),
            fps=video_details.get(cv2.CAP_PROP_FPS),
            frame_count=video_details.get(cv2.CAP_PROP_FRAME_COUNT),
            fourcc_int=video_details.get(cv2.CAP_PROP_FOURCC)
        )
        video_details.release()

        if use_gpu:
            if use_cvcuda:
                self.video_reader = cv2.cudacodec.createVideoReader(
                    filename=input_path
                )
                self.video_reader.set(colorFormat=cv2.cudacodec.ColorFormat_BGR)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]="hwaccel;cuvid|video_codec;h264_cuvid"
                self.video_reader = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG, (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY))
        else:
            self.video_reader = cv2.VideoCapture(filename=input_path, apiPreference=cv2.CAP_FFMPEG)

        self.queue_frames = queue_frames

        if self.queue_frames:
            self.video_queue = vc_queue.EagerVideoCapture(input_path, video_reader=self.video_reader, queue_size=queue_size)
        else:
            self.video_queue = None

    def is_opened(self):
        if isinstance(self.video_reader, cv2.VideoCapture):
            return self.video_reader.isOpened()
        return True

    def close(self):
        if isinstance(self.video_reader, cv2.VideoCapture):
            self.video_reader.release()

    def write_frame_by_timestamp(
            self,
            timestamp,
            path
    ):
        image=self.get_frame_by_timestamp(timestamp)
        cv_utils.core.write_image(
            image=image,
            path=path
        )

    def write_frame_by_frame_number(
            self,
            frame_number,
            path
    ):
        image = self.get_frame_by_frame_number(frame_number)
        cv_utils.core.write_image(
            image=image,
            path=path
        )

    def write_frame_by_milliseconds(
            self,
            milliseconds,
            path
    ):
        image = self.get_frame_by_milliseconds(milliseconds)
        cv_utils.core.write_image(
            image=image,
            path=path
        )

    def get_frame_by_timestamp(self, timestamp):
        if self.video_parameters.start_time is None or self.video_parameters.fps is None or self.video_parameters.frame_count is None:
            raise ValueError('Valid video start time, FPS, and frame count required to get frame by timestamp')
        try:
            timestamp = pd.to_datetime(timestamp, utc=True).to_pydatetime()
        except Exception as e:
            raise ValueError('Cannot parse start time: {}'.format(timestamp))
        frame_number = round(
            (timestamp - self.video_parameters.start_time).total_seconds()*
            self.video_parameters.fps
        )
        logger.debug('Target timestamp is {}. Selected frame number {} at timestamp {}'.format(
            timestamp.isoformat(),
            frame_number,
            (self.video_parameters.start_time + datetime.timedelta(seconds=frame_number/self.video_parameters.fps)).isoformat()
        ))
        if frame_number < 0 or frame_number > self.video_parameters.frame_count:
            raise ValueError('Specified datetime is outside the time range of the video')
        return self.get_frame_by_frame_number(frame_number)

    def get_frame_by_frame_number(self, frame_number):
        self.video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return self.get_frame()

    def get_frame_by_milliseconds(self, milliseconds):
        self.video_reader.set(cv2.CAP_PROP_POS_MSEC, milliseconds)
        return self.get_frame()

    def get_frame(self):
        success, frame = False, None
        if self.queue_frames:
            # Wait to start queue until the first frame is needed. This is in case
            # additional VideoCaptureProperties need to be applied before the
            # video begins streaming (i.e. cv2.CAP_PROP_POS_FRAMES - start frame)
            self.video_queue.start()
            success, frame = self.video_queue.read()
        else:
            if isinstance(self.video_reader, cv2.VideoCapture):
                (success, frame) = self.video_reader.read()
            elif isinstance(self.video_reader, cv2.cudacodec.VideoReader):
                (success, frame) = self.video_reader.nextFrame()

        if success:
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
        self.writer_object = cv2.VideoWriter(
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
    return cv2.VideoWriter_fourcc(*fourcc_string)


def fourcc_int_to_string(fourcc_int):
    return "".join([chr((int(fourcc_int) >> 8 * i) & 0xFF) for i in range(4)])
