import io
import time
import threading
import picamera
import numpy as np
import cv2
import os
import tkinter as tk

from camera_calibrator import OpenCVCalibrationNode
from calibrator import ChessboardInfo, Patterns
from collections import deque


class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    # print("processing image...")
                    self.stream.seek(0)
                    data = np.frombuffer(self.stream.getvalue(), dtype=np.uint8)
                    # full resolution
                    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                    # put the image in the queue
                    self.owner.q_mono.append(img)
                    # if self.owner.count > 1000:
                    #     self.owner.done = True
                    # self.owner.count += 1
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)

class ProcessOutput(object):
    def __init__(self, width, height):
        self.done = False
        # Construct a pool of 4 image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(4)]
        self.processor = None
        self.q_mono = deque([], 1)
        self.count = 0

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    self.processor = self.pool.pop()
                else:
                    # No processors available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        # Now, empty the pool, joining each thread as we go
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                    proc.terminated = True
                    proc.join()
                except IndexError:
                    break
                    # pass # pool is empty

class ConsumerThread(threading.Thread):
    def __init__(self, queue, function, OpenCVCalibrationNode):
        threading.Thread.__init__(self)
        self.queue = queue
        self.function = function
        self.node = OpenCVCalibrationNode

    def run(self):
        while True:
            # wait for an image (could happen at the very beginning when the queue is still empty)
            while len(self.queue) == 0:
                time.sleep(0.1)
            self.function(self.node, self.queue[0])


class MainActivity(object):
    def __init__(self, camera):
        self.camera = camera

        width = 9
        height = 6

        # define checkerboard info
        boards = []
        size = [ str(width)+"x"+str(height) ]
        square = ["0.108"]
        for (sz, sq) in zip(size, square):
            size = tuple([int(c) for c in sz.split('x')])
            boards.append(ChessboardInfo(size[0], size[1], float(sq)))
        pattern = Patterns.Chessboard

        ## start up the calibration node
        self.node = OpenCVCalibrationNode(boards, pattern)

        ## set camera parameters
        # camera.resolution = (3280, 2464)
        # camera.sensor_mode = 2
        self.camera.resolution = (1640, 1232)#(760, 480)
        self.camera.framerate = 15
        
        ## start streaming
        #camera.start_preview()#resolution=(760, 480))
        time.sleep(1)
        output = ProcessOutput(width, height)
        self.camera.start_recording(output, format='mjpeg', quality=100)

        ## spin up a thread for processing images
        img_thread = ConsumerThread(output.q_mono, OpenCVCalibrationNode.handle_monocular, self.node)
        img_thread.setDaemon(True)
        img_thread.start()

        while not self.node.calibration_button_pressed:
            self.camera.wait_recording(0.1)
        
        print("processing calibration images...")

        # create a window to inform the user that we are waiting on the processor -- the window will close itself once the calibration is complete
        self.root = tk.Tk()
        self.root.title("Processing...")
        label = tk.Label(self.root, text="Please wait while the calibration images are processed.")
        label.pack()
        self.root.after(200, self.checkForCalComplete)
        self.root.mainloop()

        print("rms: ", self.node.c.rms)
        
        # wait until the user presses "save"
        while not self.node.c.saved:
            self.camera.wait_recording(0.1)

        print("parameters saved!")

        self.camera.wait_recording(1)

        #capture
        myStreamFull = io.BytesIO()
        self.camera.capture(myStreamFull, format='jpeg')
        dataFull = np.frombuffer(myStreamFull.getvalue(), dtype=np.uint8)
        imgFull = cv2.imdecode(dataFull, 1)

        cv2.imshow('Raw image', imgFull)
        #cv2.imwrite('raw.jpg', imgFull)

        #undistort
        img_undistorted = cv2.undistort(imgFull, self.node.c.intrinsics, self.node.c.distortion)
        cv2.imshow('Undistorted', img_undistorted)
        #cv2.imwrite('undistor.jpg', img_undistorted)
        cv2.waitKey(0)

        output.done = True
        self.camera.stop_recording()
        img_thread.terminated = True
        img_thread.join()
        print("process complete!")


    def checkForCalComplete(self):
        # wait to process so we can print the calculated rms
        while not self.node.c.calibrated:
            self.camera.wait_recording(0.1)
        self.root.destroy()


if __name__ == '__main__':
    print("starting camera...")
    with picamera.PiCamera() as camera:
        MainActivity(camera)

    print("shutting down")