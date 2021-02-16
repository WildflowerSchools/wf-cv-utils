# cv_utils

Miscellaneous utilities for working with camera data:

* Utilities for working with images and videos (many of which use OpenCV "under the hood")

* Functions which support basic visualization of image and object points

* Functions which support camera calibration using OpenCV and/or COLMAP


## Task list

* Update dependencies
* Clean up API (rationalize all of the `__init__.py` files, etc.)
* Clean up internal cross-module references
* Add option to `project_points()` to include points that are projected inside the frame with distortion but just outside the frame without
* When API for specifying video start times changes in `wf-video-io`, update here
* Create endpoint for preparing COLMAP inputs
* Create endpoint for processing COLMAP outputs
* Add error checking to COLMAP calibration `bash` script
* Add more options to COLMAP calibration `bash` script
* Integrate Docker-based COLMAP processing
* Integrate (Docker-based?) version of OpenCV which supports proprietary codecs (primarily `mp4`)
* Consider implementing additional objects/classes (e.g., `Camera`)
* Move Honeycomb functionality to separate package
* Move Wildflower-specific functionality into separate package
* Add additional functions so we can eliminate `OpenCV` as dependency for `wf-process-pose_data` and `wf-video-io`
* Eliminate circular dependency between `wf-cv-utils` and `wf-video-io`
* Implement more performant replacement for `cv.triangulatePoints()`
* Augment drawing functions so they can handle multiple objects, `Nan` values, etc.
* Figure out what to do with objects in `calibration.geom`
* Get ride of unused/buggy functions in `core`
* Clean up color conversion helper functions (use OpenCV functions?)
* Move 3D projection code into its own submodule
* Convert drawing functions in `core` to object-oriented Matplotlib interface?
* Convert drawing functions in `core` to OpenCV drawing API?
* Fix comments in `generate_camera_pose()` (currently describes yaw inaccurately)
* Clean up handling of coordinates (shouldn't OpenCV accept numpy arrays?)
* Clean up handling of large integer coordinates
