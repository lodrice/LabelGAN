# coding: utf-8

# # Tensorflow Diffeomorphism
#
# A tensorflow implementation of [diffeomorphism](https://en.wikipedia.org/wiki/Diffeomorphism) image transformation.
#
# This implementation uses tensorflow's code for the spatial transformer networks. See https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py
#
#


from skimage.transform import warp, resize, rescale
from skimage.filters import gaussian
from skimage.color import rgb2gray

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import tensorflow as tf


# Copied Tensorflow Code


def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

def _interpolate(im, x, y, out_size):
    with tf.variable_scope('_interpolate'):
        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        return output


def _meshgrid(height, width):
    with tf.variable_scope('_meshgrid'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
        return grid


def _meshgrid_2d(height, width):
    with tf.variable_scope('_meshgrid2d'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))
        return x_t, y_t


def _transform(theta, input_dim, out_size):
    with tf.variable_scope('_transform'):
        num_batch = tf.shape(input_dim)[0]
        height = tf.shape(input_dim)[1]
        width = tf.shape(input_dim)[2]
        num_channels = tf.shape(input_dim)[3]
        theta = tf.reshape(theta, (-1, 2, 3))
        theta = tf.cast(theta, 'float32')

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width)
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([num_batch]))
        grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(theta, grid)
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        x_s_flat = x_s_flat[::-1]
        y_s_flat = tf.reshape(y_s, [-1])

        input_transformed = _interpolate(
            input_dim, x_s_flat, y_s_flat,
            out_size)

        output = tf.reshape(
            input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
        return output


def tf_warp(im, x, y):
    """
    Warp the ``im`` tensor. ``x`` and ``y`` are of scape ``[batch_size, num_pixels]``
    where ``num_pixels`` is normally height*width.
    """
    with tf.variable_scope('warp'):

        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        num_channels = tf.shape(im)[3]

        out_height = height
        out_width = width

        out_size = np.stack([out_height, out_width, num_channels])
        x_s_flat = tf.reshape(x, [-1])
        y_s_flat = tf.reshape(y, [-1])

        input_transformed = _interpolate(
            im, x_s_flat, y_s_flat,
            out_size)
        output = tf.reshape(
            input_transformed,
            tf.stack([num_batch, out_height, out_width, num_channels]))
        return output


def _arange_repeat(to, n_repeats):
    """Helper function. For example, returns [0 0 1 1 2 2] for _arange_repeat(to=3, n_repeats=2)."""
    idx = tf.range(to)
    idx = tf.reshape(idx, [-1, 1])      # Convert to a len(yp) x 1 matrix.
    idx = tf.tile(idx, tf.cast(tf.stack([1, n_repeats]), 'int32'))  # Create multiple columns.
    idx = tf.reshape(idx, [-1])         # Convert back to a vector.
    return idx


def tf_diffeomorphism(input, diffeomorphism_map):
    """
    Applies a diffeomorphis given in ``diffeomorphism_map`` to the ``input`` tensor.

    Args:
        input (tensor 4d): image tensor of shape (batch_size, height, width, channels)
        diffeomorphism_map (tensor 4d): has shape (batch_size, diff_height, diff_width, 2).
            The last axis represent the translation in y and x direction.
            The coordinate system used here is [-1, 1] x [-1, 1].
            E.g. dx = 0.5 corresponds to move the pixels by a quarter of the
            image width to the right.

    """
    def to_int_tiled(x, max_value):
        x_scaled = 0.5*(x + 1) * tf.to_float(width)
        x_int = tf.cast(tf.floor(x_scaled), 'int32')
        x_int = tf.clip_by_value(x_int, 0, width - 1)
        return tf.tile(x_int, [batch_size])

    with tf.variable_scope('diffeomorphism'):
        batch_size, height, width = input.shape[:3]
        batch_size = tf.to_int32(batch_size)

        diff_resized = tf.image.resize_images(
            diffeomorphism_map,
            size=input.shape[1:3],
            method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

        x_2d, y_2d = _meshgrid_2d(height, width)
        x_flat = tf.reshape(x_2d, [-1])
        y_flat = tf.reshape(y_2d, [-1])
        
        x_f_tiled = tf.tile(x_flat, [batch_size])
        y_f_tiled = tf.tile(y_flat, [batch_size])
        
        x_int_tiled = to_int_tiled(x_flat, width - 1)
        y_int_tiled = to_int_tiled(y_flat, height - 1)

        batch_indicies = _arange_repeat(batch_size, tf.to_int64(height*width))

        indicies = tf.stack([batch_indicies, y_int_tiled, x_int_tiled])

        diff_resized_x = diff_resized[:, :, :, 1]
        diff_resized_y = diff_resized[:, :, :, 0]

        x_off = tf.gather_nd(diff_resized_x, tf.transpose(indicies))
        y_off = tf.gather_nd(diff_resized_y, tf.transpose(indicies))
        return tf_warp(input, x_f_tiled - x_off, y_f_tiled - y_off)


def combine_diffeomorphism(diffeomorphisms, size=None):
    if size is None:
        shapes = tf.stack([d.shape[1:3] for d in diffeomorphisms])
        size = tf.reduce_max(shapes, axis=0)

    rescaled = [tf.image.resize_images(
        d,
        size=size,
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=True) for d in diffeomorphisms]

    return sum(rescaled)


def fix_border_of_diffeomorphism(diffeomorphism, width, shape=None, sigma_blur=None):
    def fix_border_mask(shape, width, sigma_blur=None):
        """
        """
        frame = np.ones(shape)

        width = int(np.floor(max(shape) / 2 * width))
        if sigma_blur is None:
            sigma_blur = width / 2
        frame[:, :width] = 0
        frame[:, -width:] = 0
        frame[:width, :] = 0
        frame[-width:, :] = 0
        return gaussian(frame, sigma_blur, cval=1).astype(np.float32)

    diff_shape = diffeomorphism.get_shape().as_list()[1:3]
    if shape is None:
        shape = diff_shape

    if shape != diff_shape:
        diffeomorphism = tf.image.resize_images(
            diffeomorphism,
            size=shape,
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=True)

    mask = fix_border_mask(shape, width, sigma_blur)[np.newaxis, :, :, np.newaxis]
    return diffeomorphism * tf.constant(mask)


def random_uniform_diffeomorphismus(shape, intensity):
    return np.random.uniform(-intensity, intensity, shape + (2,))
