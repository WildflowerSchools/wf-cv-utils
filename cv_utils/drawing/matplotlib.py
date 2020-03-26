import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def plot_2d_image_points(
        image_points,
        image_size=None,
        point_labels=[],
        show_axes=True):
    image_points = np.asarray(image_points).reshape((-1, 2))
    draw_2d_image_points(
        image_points,
        point_labels)
    format_2d_image_plot(image_size, show_axes)
    plt.show()

def plot_3d_object_points_topdown(
        object_points,
        room_corners=None,
        point_labels=[]):
    object_points = np.asarray(object_points).reshape((-1, 3))
    draw_3d_object_points_topdown(
        object_points,
        point_labels)
    format_3d_topdown_plot(
        room_corners)
    plt.show()

# Take an image in OpenCV format and plot it as a Matplotlib plot. Calls the
# drawing function above, adds formating, and shows the plot.
def plot_background_image(
        image,
        alpha=None,
        show_axes=True):
    if alpha is None:
        alpha = 0.4
    image_size = np.array([
        image.shape[1],
        image.shape[0]])
    draw_background_image(image, alpha)
    format_2d_image_plot(image_size, show_axes)
    plt.show()

def draw_2d_image_points(
        image_points,
        point_labels=[]):
    image_points = np.asarray(image_points).reshape((-1, 2))
    points_image_u = image_points[:, 0]
    points_image_v = image_points[:, 1]
    plt.plot(
        points_image_u,
        points_image_v,
        '.')
    if len(point_labels) > 0:
        for i in range(len(point_labels)):
            plt.text(points_image_u[i], points_image_v[i], point_labels[i])

def draw_3d_object_points_topdown(
        object_points,
        point_labels=[]):
    object_points = np.asarray(object_points).reshape((-1, 3))
    points_x = object_points[:, 0]
    points_y = object_points[:, 1]
    plt.plot(
        points_x,
        points_y,
        '.')
    if len(point_labels) > 0:
        for i in range(len(point_labels)):
            plt.text(points_x[i], points_y[i], point_labels[i])

# Take an image in OpenCV format and draw it as a background for a Matplotlib
# plot. We separate this from the plotting function below because we might want
# to draw other elements before formatting and showing the chart.
def draw_background_image(
        image,
        alpha=None):
    if alpha is None:
        alpha = 0.4
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB), alpha=alpha)

def format_2d_image_plot(
        image_size=None,
        show_axes=True):
    if image_size is not None:
        plt.xlim(0, image_size[0])
        plt.ylim(0, image_size[1])
    if show_axes:
        plt.xlabel(r'$u$')
        plt.ylabel(r'$v$')
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
    else:
        plt.axis('off')
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')

def format_3d_topdown_plot(
        room_corners=None):
    if room_corners is not None:
        plt.xlim(room_corners[0][0], room_corners[1][0])
        plt.ylim(room_corners[0][1], room_corners[1][1])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.gca().set_aspect('equal')
