import numpy as np
import cv2
import threading
import time

class VideoReader():
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(self.video_path)
        self.frame_time = (1/self.vid_cap.get(cv2.CAP_PROP_FPS)) * 1000 # in ms
        self.cur_frame = np.zeros((416,416,3), np.uint8)
        self.frame_num = 0

        self.is_started = False
        self.frame_lock = threading.Lock()

        self.read_thread = threading.Thread(target=self.read_thread_func)
        self.read_thread.daemon = True
        self.read_thread.start()

    def reset(self):
        self.vid_cap = cv2.VideoCapture(self.video_path)
        self.cur_frame = np.zeros((416,416,3), np.uint8)
        self.read_thread = threading.Thread(target=self.read_thread_func)
        self.read_thread.daemon = True
        self.read_thread.start()

    def read_thread_func(self):
        while True:
            if self.is_started:
                ret, frame = self.vid_cap.read()
                
                self.frame_lock.acquire()
                if ret:
                    self.cur_frame = frame.copy()
                # Video has finished being read
                else: 
                    self.cur_frame = None
                self.frame_lock.release()

                # End thread
                if frame is None:
                    break

                time.sleep(self.frame_time/1000)    
                #cv2.imshow("hmm", frame)
                #cv2.waitKey(int(self.frame_time))
    
    def read(self):
        if self.frame_num >= 1: # Needed because the very first detection frame takes its own sweet time
            self.is_started = True
    
        frame = None
        
        self.frame_lock.acquire()
        if self.cur_frame is not None:
            frame = self.cur_frame.copy()
        self.frame_lock.release()
        
        self.frame_num += 1

        return None, self.cur_frame
