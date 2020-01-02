# Task list

* Clean up color conversion helper functions (use OpenCV functions?)
* Move Wildflower-specific code into separate package
* Remove unused code in `core`
* Move 3D projection code into its own submodule
* Convert drawing functions in `core` to object-oriented Matplotlib interface?
* Convert drawing functions in `core` to OpenCV drawing API?
* Fix comments in `generate_camera_pose()` (currently describes yaw inaccurately)
* Clean up handling of coordinates (shouldn't OpenCV accept numpy arrays?)
* Clean up handling of large integer coordinates
* Figure out how to put codec in the right system path
* Check container of video output (does it match input or is it AVI?)
* Add camera object which contains calibration data (and ID?)
