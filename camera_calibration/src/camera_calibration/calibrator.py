#!/usr/bin/env python
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2009, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the Willow Garage nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
from io import BytesIO
import cv2
#import cv_bridge
import math
import numpy.linalg
import pickle
import random
import tarfile
import time
from distutils.version import LooseVersion


# Supported calibration patterns
class Patterns:
    Chessboard, Circles, ACircles = list(range(3))

class CalibrationException(Exception):
    pass

# TODO: Make pattern per-board?
class ChessboardInfo(object):
    def __init__(self, n_cols = 0, n_rows = 0, dim = 0.0):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.dim = dim

# Make all private!!!!!
def lmin(seq1, seq2):
    """ Pairwise minimum of two sequences """
    return [min(a, b) for (a, b) in zip(seq1, seq2)]

def lmax(seq1, seq2):
    """ Pairwise maximum of two sequences """
    return [max(a, b) for (a, b) in zip(seq1, seq2)]

def _pdist(p1, p2):
    """
    Distance bwt two points. p1 = (x, y), p2 = (x, y)
    """
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

def _get_outside_corners(corners, board):
    """
    Return the four corners of the board as a whole, as (up_left, up_right, down_right, down_left).
    """
    xdim = board.n_cols
    ydim = board.n_rows

    if corners.shape[1] * corners.shape[0] != xdim * ydim:
        raise Exception("Invalid number of corners! %d corners. X: %d, Y: %d" % (corners.shape[1] * corners.shape[0],
                                                                                 xdim, ydim))

    up_left    = corners[0,0]
    up_right   = corners[xdim - 1,0]
    down_right = corners[-1,0]
    down_left  = corners[-xdim,0]

    return (up_left, up_right, down_right, down_left)

def _get_skew(corners, board):
    """
    Get skew for given checkerboard detection.
    Scaled to [0,1], which 0 = no skew, 1 = high skew
    Skew is proportional to the divergence of three outside corners from 90 degrees.
    """
    # TODO Using three nearby interior corners might be more robust, outside corners occasionally
    # get mis-detected
    up_left, up_right, down_right, _ = _get_outside_corners(corners, board)

    def angle(a, b, c):
        """
        Return angle between lines ab, bc
        """
        ab = a - b
        cb = c - b
        return math.acos(numpy.dot(ab,cb) / (numpy.linalg.norm(ab) * numpy.linalg.norm(cb)))

    skew = min(1.0, 2. * abs((math.pi / 2.) - angle(up_left, up_right, down_right)))
    return skew

def _get_area(corners, board):
    """
    Get 2d image area of the detected checkerboard.
    The projected checkerboard is assumed to be a convex quadrilateral, and the area computed as
    |p X q|/2; see http://mathworld.wolfram.com/Quadrilateral.html.
    """
    (up_left, up_right, down_right, down_left) = _get_outside_corners(corners, board)
    a = up_right - up_left
    b = down_right - up_right
    c = down_left - down_right
    p = b + c
    q = a + b
    return abs(p[0]*q[1] - p[1]*q[0]) / 2.

def _get_corners(img, board, refine = True, checkerboard_flags=0):
    """
    Get corners for a particular chessboard for an image
    """
    h = img.shape[0]
    w = img.shape[1]
    if len(img.shape) == 3 and img.shape[2] == 3:
        mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        mono = img
    (ok, corners) = cv2.findChessboardCorners(mono, (board.n_cols, board.n_rows), flags = cv2.CALIB_CB_ADAPTIVE_THRESH |
                                              cv2.CALIB_CB_NORMALIZE_IMAGE | checkerboard_flags)
    if not ok:
        return (ok, corners)

    # If any corners are within BORDER pixels of the screen edge, reject the detection by setting ok to false
    # NOTE: This may cause problems with very low-resolution cameras, where 8 pixels is a non-negligible fraction
    # of the image size. See http://answers.ros.org/question/3155/how-can-i-calibrate-low-resolution-cameras
    BORDER = 8
    if not all([(BORDER < corners[i, 0, 0] < (w - BORDER)) and (BORDER < corners[i, 0, 1] < (h - BORDER)) for i in range(corners.shape[0])]):
        ok = False

    # Ensure that all corner-arrays are going from top to bottom.
    if board.n_rows!=board.n_cols:
        if corners[0, 0, 1] > corners[-1, 0, 1]:
            corners = numpy.copy(numpy.flipud(corners))
    else:
        direction_corners=(corners[-1]-corners[0])>=numpy.array([[0.0,0.0]])
    
        if not numpy.all(direction_corners):
            if not numpy.any(direction_corners):
                corners = numpy.copy(numpy.flipud(corners))
            elif direction_corners[0][0]:
                corners=numpy.rot90(corners.reshape(board.n_rows,board.n_cols,2)).reshape(board.n_cols*board.n_rows,1,2)
            else:
                corners=numpy.rot90(corners.reshape(board.n_rows,board.n_cols,2),3).reshape(board.n_cols*board.n_rows,1,2)

    if refine and ok:
        # Use a radius of half the minimum distance between corners. This should be large enough to snap to the
        # correct corner, but not so large as to include a wrong corner in the search window.
        min_distance = float("inf")
        for row in range(board.n_rows):
            for col in range(board.n_cols - 1):
                index = row*board.n_rows + col
                min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + 1, 0]))
        for row in range(board.n_rows - 1):
            for col in range(board.n_cols):
                index = row*board.n_rows + col
                min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + board.n_cols, 0]))
        radius = int(math.ceil(min_distance * 0.5))
        cv2.cornerSubPix(mono, corners, (radius,radius), (-1,-1),
                                      ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1 ))

    return (ok, corners)

def _get_circles(img, board, pattern):
    """
    Get circle centers for a symmetric or asymmetric grid
    """
    h = img.shape[0]
    w = img.shape[1]
    if len(img.shape) == 3 and img.shape[2] == 3:
        mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        mono = img

    flag = cv2.CALIB_CB_SYMMETRIC_GRID
    if pattern == Patterns.ACircles:
        flag = cv2.CALIB_CB_ASYMMETRIC_GRID
    mono_arr = numpy.array(mono)
    (ok, corners) = cv2.findCirclesGrid(mono_arr, (board.n_cols, board.n_rows), flags=flag)

    # In symmetric case, findCirclesGrid does not detect the target if it's turned sideways. So we try
    # again with dimensions swapped - not so efficient.
    # TODO Better to add as second board? Corner ordering will change.
    if not ok and pattern == Patterns.Circles:
        (ok, corners) = cv2.findCirclesGrid(mono_arr, (board.n_rows, board.n_cols), flags=flag)

    return (ok, corners)


# TODO self.size needs to come from CameraInfo, full resolution
class Calibrator(object):
    """
    Base class for calibration system
    """
    def __init__(self, boards, flags=0, pattern=Patterns.Chessboard, name='', 
    checkerboard_flags=cv2.CALIB_CB_FAST_CHECK, max_chessboard_speed = -1.0):

        # Set to true after we perform calibration
        self.calibrated = False
        self.calib_flags = flags
        self.checkerboard_flags = checkerboard_flags
        self.pattern = pattern
        self._boards = [ChessboardInfo(max(i.n_cols, i.n_rows), min(i.n_cols, i.n_rows), i.dim) for i in boards]
        # self.br = cv_bridge.CvBridge()
        
        # set to true after saving
        self.saved = False

        # self.db is list of (parameters, image) samples for use in calibration. parameters has form
        # (X, Y, size, skew) all normalized to [0,1], to keep track of what sort of samples we've taken
        # and ensure enough variety.
        self.db = []
        # For each db sample, we also record the detected corners.
        self.good_corners = []
        # Set to true when we have sufficiently varied samples to calibrate
        self.goodenough = False
        self.param_ranges = [0.7, 0.7, 0.4, 0.5]
        self.name = name
        self.last_frame_corners = None
        self.max_chessboard_speed = max_chessboard_speed

    def get_parameters(self, corners, board, size):
        """
        Return list of parameters [X, Y, size, skew] describing the checkerboard view.
        """
        (width, height) = size
        Xs = corners[:,:,0]
        Ys = corners[:,:,1]
        area = _get_area(corners, board)
        border = math.sqrt(area)
        # For X and Y, we "shrink" the image all around by approx. half the board size.
        # Otherwise large boards are penalized because you can't get much X/Y variation.
        p_x = min(1.0, max(0.0, (numpy.mean(Xs) - border / 2) / (width  - border)))
        p_y = min(1.0, max(0.0, (numpy.mean(Ys) - border / 2) / (height - border)))
        p_size = math.sqrt(area / (width * height))
        skew = _get_skew(corners, board)
        params = [p_x, p_y, p_size, skew]
        return params

    def is_slow_moving(self, corners, last_frame_corners):
        """
        Returns true if the motion of the checkerboard is sufficiently low between
        this and the previous frame.
        """
        # If we don't have previous frame corners, we can't accept the sample
        if last_frame_corners is None:
            return False
        num_corners = len(corners)
        corner_deltas = (corners - last_frame_corners).reshape(num_corners, 2)
        # Average distance travelled overall for all corners
        average_motion = numpy.average(numpy.linalg.norm(corner_deltas, axis = 1))
        return average_motion <= self.max_chessboard_speed

    def is_good_sample(self, params, corners, last_frame_corners):
        """
        Returns true if the checkerboard detection described by params should be added to the database.
        """
        if not self.db:
            return True

        def param_distance(p1, p2):
            return sum([abs(a-b) for (a,b) in zip(p1, p2)])

        db_params = [sample[0] for sample in self.db]
        d = min([param_distance(params, p) for p in db_params])
        #print "d = %.3f" % d #DEBUG
        # TODO What's a good threshold here? Should it be configurable?
        if d <= 0.2:
            return False

        if self.max_chessboard_speed > 0:
            if not self.is_slow_moving(corners, last_frame_corners):
                return False

        # All tests passed
        return True

    _param_names = ["X", "Y", "Size", "Skew"]

    def compute_goodenough(self):
        if not self.db:
            return None

        # Find range of checkerboard poses covered by samples in database
        all_params = [sample[0] for sample in self.db]
        min_params = all_params[0]
        max_params = all_params[0]
        for params in all_params[1:]:
            min_params = lmin(min_params, params)
            max_params = lmax(max_params, params)
        # Don't reward small size or skew
        min_params = [min_params[0], min_params[1], 0., 0.]

        # For each parameter, judge how much progress has been made toward adequate variation
        progress = [min((hi - lo) / r, 1.0) for (lo, hi, r) in zip(min_params, max_params, self.param_ranges)]
        # If we have lots of samples, allow calibration even if not all parameters are green
        # TODO Awkward that we update self.goodenough instead of returning it
        self.goodenough = (len(self.db) >= 40) or all([p == 1.0 for p in progress])

        return list(zip(self._param_names, min_params, max_params, progress))

    def mk_object_points(self, boards, use_board_size = False):
        opts = []
        for i, b in enumerate(boards):
            num_pts = b.n_cols * b.n_rows
            opts_loc = numpy.zeros((num_pts, 1, 3), numpy.float32)
            for j in range(num_pts):
                opts_loc[j, 0, 0] = (j / b.n_cols)
                if self.pattern == Patterns.ACircles:
                    opts_loc[j, 0, 1] = 2*(j % b.n_cols) + (opts_loc[j, 0, 0] % 2)
                else:
                    opts_loc[j, 0, 1] = (j % b.n_cols)
                opts_loc[j, 0, 2] = 0
                if use_board_size:
                    opts_loc[j, 0, :] = opts_loc[j, 0, :] * b.dim
            opts.append(opts_loc)
        return opts

    def get_corners(self, img, refine = True):
        """
        Use cvFindChessboardCorners to find corners of chessboard in image.

        Check all boards. Return corners for first chessboard that it detects
        if given multiple size chessboards.

        Returns (ok, corners, board)
        """

        for b in self._boards:
            if self.pattern == Patterns.Chessboard:
                (ok, corners) = _get_corners(img, b, refine, self.checkerboard_flags)
            else:
                (ok, corners) = _get_circles(img, b, self.pattern)
            if ok:
                return (ok, corners, b)
        return (False, None, None)

    def downsample_and_detect(self, img):
        """
        Downsample the input image to approximately VGA resolution and detect the
        calibration target corners in the full-size image.

        Combines these apparently orthogonal duties as an optimization. Checkerboard
        detection is too expensive on large images, so it's better to do detection on
        the smaller display image and scale the corners back up to the correct size.

        Returns (scrib, corners, downsampled_corners, board, (x_scale, y_scale)).
        """
        # Scale the input image down to ~VGA size
        height = img.shape[0]
        width = img.shape[1]
        scale = math.sqrt( (width*height) / (640.*480.) )
        if scale > 1.0:
            scrib = cv2.resize(img, (int(width / scale), int(height / scale)))
        else:
            scrib = img
        # Due to rounding, actual horizontal/vertical scaling may differ slightly
        x_scale = float(width) / scrib.shape[1]
        y_scale = float(height) / scrib.shape[0]

        if self.pattern == Patterns.Chessboard:
            # Detect checkerboard
            (ok, downsampled_corners, board) = self.get_corners(scrib, refine = True)

            # Scale corners back to full size image
            corners = None
            if ok:
                if scale > 1.0:
                    # Refine up-scaled corners in the original full-res image
                    # TODO Does this really make a difference in practice?
                    corners_unrefined = downsampled_corners.copy()
                    corners_unrefined[:, :, 0] *= x_scale
                    corners_unrefined[:, :, 1] *= y_scale
                    radius = int(math.ceil(scale))
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        mono = img
                    cv2.cornerSubPix(mono, corners_unrefined, (radius,radius), (-1,-1),
                                                  ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1 ))
                    corners = corners_unrefined
                else:
                    corners = downsampled_corners
        else:
            # Circle grid detection is fast even on large images
            (ok, corners, board) = self.get_corners(img)
            # Scale corners to downsampled image for display
            downsampled_corners = None
            if ok:
                if scale > 1.0:
                    downsampled_corners = corners.copy()
                    downsampled_corners[:,:,0] /= x_scale
                    downsampled_corners[:,:,1] /= y_scale
                else:
                    downsampled_corners = corners

        return (scrib, corners, downsampled_corners, board, (x_scale, y_scale))


    def lrmsg(self, d, k, r, p):
        """ Used by :meth:`as_message`.  Return a CameraInfo message for the given calibration matrices """
        msg = sensor_msgs.msg.CameraInfo()
        (msg.width, msg.height) = self.size
        if d.size > 5:
            msg.distortion_model = "rational_polynomial"
        else:
            msg.distortion_model = "plumb_bob"
        msg.D = numpy.ravel(d).copy().tolist()
        msg.K = numpy.ravel(k).copy().tolist()
        msg.R = numpy.ravel(r).copy().tolist()
        msg.P = numpy.ravel(p).copy().tolist()
        return msg

    def lrreport(self, d, k, r, p):
        print("D = ", numpy.ravel(d).tolist())
        print("K = ", numpy.ravel(k).tolist())
        print("R = ", numpy.ravel(r).tolist())
        print("P = ", numpy.ravel(p).tolist())

    def lrost(self, name, d, k, r, p):
        calmessage = (
        "# oST version 5.0 parameters\n"
        + "\n"
        + "\n"
        + "[image]\n"
        + "\n"
        + "width\n"
        + str(self.size[0]) + "\n"
        + "\n"
        + "height\n"
        + str(self.size[1]) + "\n"
        + "\n"
        + "[%s]" % name + "\n"
        + "\n"
        + "camera matrix\n"
        + " ".join(["%8f" % k[0,i] for i in range(3)]) + "\n"
        + " ".join(["%8f" % k[1,i] for i in range(3)]) + "\n"
        + " ".join(["%8f" % k[2,i] for i in range(3)]) + "\n"
        + "\n"
        + "distortion\n"
        + " ".join(["%8f" % d[i,0] for i in range(d.shape[0])]) + "\n"
        + "\n"
        + "rectification\n"
        + " ".join(["%8f" % r[0,i] for i in range(3)]) + "\n"
        + " ".join(["%8f" % r[1,i] for i in range(3)]) + "\n"
        + " ".join(["%8f" % r[2,i] for i in range(3)]) + "\n"
        + "\n"
        + "projection\n"
        + " ".join(["%8f" % p[0,i] for i in range(4)]) + "\n"
        + " ".join(["%8f" % p[1,i] for i in range(4)]) + "\n"
        + " ".join(["%8f" % p[2,i] for i in range(4)]) + "\n"
        + "\n")
        assert len(calmessage) < 525, "Calibration info must be less than 525 bytes"
        return calmessage

    def lryaml(self, name, d, k, r, p):
        calmessage = (""
        + "image_width: " + str(self.size[0]) + "\n"
        + "image_height: " + str(self.size[1]) + "\n"
        + "camera_name: " + name + "\n"
        + "camera_matrix:\n"
        + "  rows: 3\n"
        + "  cols: 3\n"
        + "  data: [" + ", ".join(["%8f" % i for i in k.reshape(1,9)[0]]) + "]\n"
        + "distortion_model: " + ("rational_polynomial" if d.size > 5 else "plumb_bob") + "\n"
        + "distortion_coefficients:\n"
        + "  rows: 1\n"
        + "  cols: 5\n"
        + "  data: [" + ", ".join(["%8f" % d[i,0] for i in range(d.shape[0])]) + "]\n"
        + "rectification_matrix:\n"
        + "  rows: 3\n"
        + "  cols: 3\n"
        + "  data: [" + ", ".join(["%8f" % i for i in r.reshape(1,9)[0]]) + "]\n"
        + "projection_matrix:\n"
        + "  rows: 3\n"
        + "  cols: 4\n"
        + "  data: [" + ", ".join(["%8f" % i for i in p.reshape(1,12)[0]]) + "]\n"
        + "")
        return calmessage

    def do_save(self):
        # filename = '/tmp/calibrationdata.tar.gz'
        # tf = tarfile.open(filename, 'w:gz')
        # self.do_tarfile_save(tf) # Must be overridden in subclasses
        # tf.close()

        filename = "calibration_parameters"
        f = open(filename, "w")
        f.write( self.lryaml(self.name, self.distortion, self.intrinsics, self.R, self.P) )
        f.close()
        print(("Wrote calibration data to", filename))
        self.saved = True

def image_from_archive(archive, name):
    """
    Load image PGM file from tar archive. 

    Used for tarfile loading and unit test.
    """
    member = archive.getmember(name)
    imagefiledata = numpy.fromstring(archive.extractfile(member).read(), numpy.uint8)
    imagefiledata.resize((1, imagefiledata.size))
    return cv2.imdecode(imagefiledata, cv2.IMREAD_COLOR)

class ImageDrawable(object):
    """
    Passed to CalibrationNode after image handled. Allows plotting of images
    with detected corner points
    """
    def __init__(self):
        self.params = None

class MonoDrawable(ImageDrawable):
    def __init__(self):
        ImageDrawable.__init__(self)
        self.scrib = None
        self.linear_error = -1.0
                



class MonoCalibrator(Calibrator):
    """
    Calibration class for monocular cameras::

        images = [cv2.imread("mono%d.png") for i in range(8)]
        mc = MonoCalibrator()
        mc.cal(images)
        print mc.as_message()
    """

    is_mono = True  # TODO Could get rid of is_mono

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'narrow_stereo/left'
        super(MonoCalibrator, self).__init__(*args, **kwargs)

    def cal(self, images):
        """
        Calibrate camera from given images
        """
        goodcorners = self.collect_corners(images)
        self.cal_fromcorners(goodcorners)
        self.calibrated = True

    def collect_corners(self, images):
        """
        :param images: source images containing chessboards
        :type images: list of :class:`cvMat`

        Find chessboards in all images.

        Return [ (corners, ChessboardInfo) ]
        """
        self.size = (images[0].shape[1], images[0].shape[0])
        corners = [self.get_corners(i) for i in images]

        goodcorners = [(co, b) for (ok, co, b) in corners if ok]
        if not goodcorners:
            raise CalibrationException("No corners found in images!")
        return goodcorners

    def cal_fromcorners(self, good):
        """
        :param good: Good corner positions and boards 
        :type good: [(corners, ChessboardInfo)]

        
        """
        boards = [ b for (_, b) in good ]

        ipts = [ points for (points, _) in good ]
        opts = self.mk_object_points(boards)

        self.intrinsics = numpy.zeros((3, 3), numpy.float64)
        if self.calib_flags & cv2.CALIB_RATIONAL_MODEL:
            self.distortion = numpy.zeros((8, 1), numpy.float64) # rational polynomial
        else:
            self.distortion = numpy.zeros((5, 1), numpy.float64) # plumb bob
        # If FIX_ASPECT_RATIO flag set, enforce focal lengths have 1/1 ratio
        self.intrinsics[0,0] = 1.0
        self.intrinsics[1,1] = 1.0
        self.rms, self.camera_matrix, self.dist_coefs, self.rvecs, self.tvecs = cv2.calibrateCamera(
                   opts, ipts,
                   self.size, self.intrinsics,
                   self.distortion,
                   flags = self.calib_flags)

        # R is identity matrix for monocular calibration
        self.R = numpy.eye(3, dtype=numpy.float64)
        self.P = numpy.zeros((3, 4), dtype=numpy.float64)

        self.set_alpha(0.0)

    def set_alpha(self, a):
        """
        Set the alpha value for the calibrated camera solution.  The alpha
        value is a zoom, and ranges from 0 (zoomed in, all pixels in
        calibrated image are valid) to 1 (zoomed out, all pixels in
        original image are in calibrated image).
        """

        # NOTE: Prior to Electric, this code was broken such that we never actually saved the new
        # camera matrix. In effect, this enforced P = [K|0] for monocular cameras.
        # TODO: Verify that OpenCV #1199 gets applied (improved GetOptimalNewCameraMatrix)
        ncm, _ = cv2.getOptimalNewCameraMatrix(self.intrinsics, self.distortion, self.size, a)
        for j in range(3):
            for i in range(3):
                self.P[j,i] = ncm[j, i]
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.intrinsics, self.distortion, self.R, ncm, self.size, cv2.CV_32FC1)

    def remap(self, src):
        """
        :param src: source image
        :type src: :class:`cvMat`

        Apply the post-calibration undistortion to the source image
        """
        return cv2.remap(src, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def undistort_points(self, src):
        """
        :param src: N source pixel points (u,v) as an Nx2 matrix
        :type src: :class:`cvMat`

        Apply the post-calibration undistortion to the source points
        """

        return cv2.undistortPoints(src, self.intrinsics, self.distortion, R = self.R, P = self.P)

    def from_message(self, msg, alpha = 0.0):
        """ Initialize the camera calibration from a CameraInfo message """

        self.size = (msg.width, msg.height)
        self.intrinsics = numpy.array(msg.K, dtype=numpy.float64, copy=True).reshape((3, 3))
        self.distortion = numpy.array(msg.D, dtype=numpy.float64, copy=True).reshape((len(msg.D), 1))
        self.R = numpy.array(msg.R, dtype=numpy.float64, copy=True).reshape((3, 3))
        self.P = numpy.array(msg.P, dtype=numpy.float64, copy=True).reshape((3, 4))

        self.set_alpha(0.0)

    def report(self):
        self.lrreport(self.distortion, self.intrinsics, self.R, self.P)

    def ost(self):
        return self.lrost(self.name, self.distortion, self.intrinsics, self.R, self.P)

    def yaml(self):
        return self.lryaml(self.name, self.distortion, self.intrinsics, self.R, self.P)

    def linear_error_from_image(self, image):
        """
        Detect the checkerboard and compute the linear error.
        Mainly for use in tests.
        """
        _, corners, _, board, _ = self.downsample_and_detect(image)
        if corners is None:
            return None

        undistorted = self.undistort_points(corners)
        return self.linear_error(undistorted, board)

    @staticmethod
    def linear_error(corners, b):

        """
        Returns the linear error for a set of corners detected in the unrectified image.
        """

        if corners is None:
            return None

        def pt2line(x0, y0, x1, y1, x2, y2):
            """ point is (x0, y0), line is (x1, y1, x2, y2) """
            return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        cc = b.n_cols
        cr = b.n_rows
        errors = []
        for r in range(cr):
            (x1, y1) = corners[(cc * r) + 0, 0]
            (x2, y2) = corners[(cc * r) + cc - 1, 0]
            for i in range(1, cc - 1):
                (x0, y0) = corners[(cc * r) + i, 0]
                errors.append(pt2line(x0, y0, x1, y1, x2, y2))
        if errors:
            return math.sqrt(sum([e**2 for e in errors]) / len(errors))
        else:
            return None


    def handle_msg(self, msg):
        """
        Detects the calibration target and, if found and provides enough new information,
        adds it to the sample database.

        Returns a MonoDrawable message with the display image and progress info.
        """
        # gray = self.mkgray(msg)
        gray = cv2.cvtColor(msg, cv2.COLOR_BGR2GRAY)
        linear_error = -1

        # Get display-image-to-be (scrib) and detection of the calibration target
        scrib_mono, corners, downsampled_corners, board, (x_scale, y_scale) = self.downsample_and_detect(gray)

        if self.calibrated:
            # Show rectified image
            # TODO Pull out downsampling code into function
            gray_remap = self.remap(gray)
            gray_rect = gray_remap
            if x_scale != 1.0 or y_scale != 1.0:
                gray_rect = cv2.resize(gray_remap, (scrib_mono.shape[1], scrib_mono.shape[0]))

            scrib = cv2.cvtColor(gray_rect, cv2.COLOR_GRAY2BGR)

            if corners is not None:
                # Report linear error
                undistorted = self.undistort_points(corners)
                linear_error = self.linear_error(undistorted, board)

                # Draw rectified corners
                scrib_src = undistorted.copy()
                scrib_src[:,:,0] /= x_scale
                scrib_src[:,:,1] /= y_scale
                cv2.drawChessboardCorners(scrib, (board.n_cols, board.n_rows), scrib_src, True)

        else:
            scrib = cv2.cvtColor(scrib_mono, cv2.COLOR_GRAY2BGR)
            if corners is not None:
                # Draw (potentially downsampled) corners onto display image
                cv2.drawChessboardCorners(scrib, (board.n_cols, board.n_rows), downsampled_corners, True)

                # Add sample to database only if it's sufficiently different from any previous sample.
                params = self.get_parameters(corners, board, (gray.shape[1], gray.shape[0]))
                if self.is_good_sample(params, corners, self.last_frame_corners):
                    self.db.append((params, gray))
                    self.good_corners.append((corners, board))
                    print(("*** Added sample %d, p_x = %.3f, p_y = %.3f, p_size = %.3f, skew = %.3f" % tuple([len(self.db)] + params)))
        
        self.last_frame_corners = corners
        rv = MonoDrawable()
        rv.scrib = scrib
        rv.params = self.compute_goodenough()
        rv.linear_error = linear_error
        return rv

    def do_calibration(self, dump = False):
        if not self.good_corners:
            print("**** Collecting corners for all images! ****") #DEBUG
            images = [i for (p, i) in self.db]
            self.good_corners = self.collect_corners(images)
        # Dump should only occur if user wants it
        if dump:
            pickle.dump((self.is_mono, self.size, self.good_corners),
                        open("/tmp/camera_calibration_%08x.pickle" % random.getrandbits(32), "w"))
        self.size = (self.db[0][1].shape[1], self.db[0][1].shape[0]) # TODO Needs to be set externally
        self.cal_fromcorners(self.good_corners)
        self.calibrated = True
        # DEBUG
        print((self.report()))
        print((self.ost()))

    def do_tarfile_save(self, tf):
        """ Write images and calibration solution to a tarfile object """

        def taradd(name, buf):
            if isinstance(buf, basestring):
                s = StringIO(buf)
            else:
                s = BytesIO(buf)
            ti = tarfile.TarInfo(name)
            ti.size = len(s.getvalue())
            ti.uname = 'calibrator'
            ti.mtime = int(time.time())
            tf.addfile(tarinfo=ti, fileobj=s)

        ims = [("left-%04d.png" % i, im) for i,(_, im) in enumerate(self.db)]
        for (name, im) in ims:
            taradd(name, cv2.imencode(".png", im)[1].tostring())
        taradd('ost.yaml', self.yaml())
        taradd('ost.txt', self.ost())

    def do_tarfile_calibration(self, filename):
        archive = tarfile.open(filename, 'r')

        limages = [ image_from_archive(archive, f) for f in archive.getnames() if (f.startswith('left') and (f.endswith('.pgm') or f.endswith('png'))) ]

        self.cal(limages)
