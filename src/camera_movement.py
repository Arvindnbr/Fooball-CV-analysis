import cv2
import numpy as np
import sys, os
import pickle
sys.path.append(',,/')
from utils.bbox_utils import get_distance, xy_distance


class CameraMovement:
    def __init__(self, frames) -> None:
        
        self.min_cam_distance = 5

        old_gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        feature_mask = np.zeros_like(old_gray)
        feature_mask[:,0:20] = 1
        feature_mask[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = feature_mask
        )

        self.lucas_param = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0.03)
        )

    def adjusted_positions_to_track(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_no, track in enumerate(object_tracks):
                for track_id, track_data in track.items():
                    position = track_data['bbox']
                    camera_movement = camera_movement_per_frame[frame_no]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_no][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_pkl = False, pkl_path = None):

        if read_from_pkl and pkl_path is not None and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as pkl:
                return pickle.load(pkl)
            
        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_no in range(1, len(frames)):
            grayscale = cv2.cvtColor(frames[frame_no], cv2.COLOR_BGR2GRAY)

            new_features, status, err = cv2.calcOpticalFlowPyrLK(old_gray, grayscale, old_features, None, **self.lucas_param)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                featurepoint_new = new.ravel()
                featurepoint_old = old.ravel()

                distance = get_distance(featurepoint_new, featurepoint_old)

                if distance>max_distance:
                    max_distance = distance

                    camera_movement_x, camera_movement_y= xy_distance(featurepoint_old, featurepoint_new)
            
            if max_distance > self.min_cam_distance:
                camera_movement[frame_no] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(grayscale, **self.features)

            old_gray =  grayscale.copy()
        
        if pkl_path is not None:
            with open(pkl_path, 'wb') as pkl:
                pickle.dump(camera_movement, pkl)
        return camera_movement
    
    def plot_camera_movement(self, frames, camera_movement_per_frame):

        op_frame = []

        for frame_num, frame in enumerate(frames):
            overlay = frame.copy()

            cv2.rectangle(overlay, (950,640), (1120,690),(0,10,0),cv2.FILLED)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            x , y = camera_movement_per_frame[frame_num]

            frame = cv2.putText(frame, f"Camera-X ={x:.2f}", (960,660), cv2.FONT_HERSHEY_PLAIN, 1, (25,25,255), 2)
            frame = cv2.putText(frame, f"Camera-Y ={y:.2f}", (960,678), cv2.FONT_HERSHEY_PLAIN, 1, (25,25,255), 2)

            op_frame.append(frame)
        
        return op_frame









