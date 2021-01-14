# Task list

* Add option to `project_points()` to include points that are projected inside the frame with distortion but just outside the frame without
* Move all code that references Honeycomb into a separate package (`wf-minimal-honeycomb-python` or perhaps a new package)
* Move Wildflower-specific code into separate package
* Augment drawing functions so they can handle multiple objects, `Nan` values, etc.
* Break out some of the individual pieces of `visualize_calibration()` so they can be used separately
* Add functions to output COLMAP data (positions to `ref_images.txt`, intrinsic calibration to database, etc.)
* Clean up API (rationalize all of the `__init__.py` files, etc.)
* Figure out what to do with objects in `calibration.geom`
* Get ride of unused/buggy functions in `core`
* Consider creating `Camera` class
* Clean up color conversion helper functions (use OpenCV functions?)
* Move 3D projection code into its own submodule
* Convert drawing functions in `core` to object-oriented Matplotlib interface?
* Convert drawing functions in `core` to OpenCV drawing API?
* Fix comments in `generate_camera_pose()` (currently describes yaw inaccurately)
* Clean up handling of coordinates (shouldn't OpenCV accept numpy arrays?)
* Clean up handling of large integer coordinates
* Figure out how to put codec in the right system path
* Check container of video output (does it match input or is it AVI?)
* Add camera object which contains calibration data (and ID?)
