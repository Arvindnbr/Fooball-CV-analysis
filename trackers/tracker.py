from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import numpy as np
import pandas as pd
import cv2
sys.path.append('../')
from utils import get_bbox_centre, get_bbox_width_height




class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            batch_detection = self.model.predict(frames[i:i+batch_size], conf=0.4)
            detections += batch_detection
        return detections

    def get_object_tracks(self, frames, read_from_pkl=False, pkl_path=None):

        if read_from_pkl and pkl_path is not None and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as file:
                tracks = pickle.load(file)
            return tracks

        #dictionary to store track details
        tracks = {
            "player":[],
            "football":[],
            "referee":[]
        }

        detections = self.detect_frames(frames)
        for frame_no, detection in enumerate(detections):
            class_name = detection.names
            class_name_inv = {v:k for k,v in class_name.items()}

            #convert to supervision detection format
            sv_detection = sv.Detections.from_ultralytics(detection)

            for obj_index, class_id in enumerate(sv_detection.class_id):
                if class_name[class_id] == 'goalkeeper':
                    sv_detection.class_id[obj_index] = class_name_inv["player"]

            #track detections
            detection_with_tracks = self.tracker.update_with_detections(sv_detection)

            tracks["player"].append({})
            tracks["football"].append({})
            tracks["referee"].append({})

            for frame in detection_with_tracks:
                class_id = frame[3]
                bbox = frame[0].tolist()
                track_id = frame[4]

                if class_id == class_name_inv["player"]:
                    tracks["player"][frame_no][track_id] = {"bbox": bbox}
                if class_id == class_name_inv["referee"]:
                    tracks["referee"][frame_no][track_id] = {"bbox": bbox}
            
            for frame in sv_detection:
                class_id = frame[3]
                bbox = frame[0].tolist()

                if class_id == class_name_inv["football"]:
                    tracks["football"][frame_no][1] = {"bbox": bbox}

        if pkl_path is not None:
            with open(pkl_path, 'wb') as file:
                pickle.dump(tracks, file)
            
        return tracks
    
    def interpolate_ball_tracks(self, ball_positions):

        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        #interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def draw_triangle(self, frames, bbox, color):

        y = int(bbox[1])
        x, _ = get_bbox_centre(bbox)
        traingle = np.array([
           [x,y],
           [x-10, y-20],
           [x+10, y-20] 
        ])

        cv2.drawContours(frames, [traingle], 0,color, cv2.FILLED)
        cv2.drawContours(frames, [traingle], 0, (255,255,255), 2)
        return frames


    

    def draw_ellipse(self, frames, bbox, color, track_id = None):
        x_centre, y_centre = get_bbox_centre(bbox)
        y2 = bbox[3]
        width, height= get_bbox_width_height(bbox)
        cv2.ellipse(frames,
            center=(x_centre, int(y2)),
            axes=(int(width), int(0.2*int(width))),
            angle=0.0,
            startAngle=240,#240
            endAngle=-60,#-60
            color=color,
            thickness=2,
            lineType= cv2.LINE_AA
            )
        
        rect_width = 35
        rect_height = 20   
        x1rect = x_centre - rect_width//2
        y1rect = (int(y2) - rect_height//2) + 15
        x2rect = x_centre + rect_width//2
        y2rect = (int(y2) + rect_height//2) + 15 

        if track_id is not None:
            cv2.rectangle(frames,(x1rect,y1rect),(x2rect,y2rect),color, cv2.FILLED)

            x1txt = x1rect + 12
            if track_id>99:
                x1txt -=3
            if track_id>9:
                x1txt -=5

            cv2.putText(frames, str(track_id),(x1txt,y2rect-5 ), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0), 2 )


        return frames
    
    def plot_annotations(self, frames,tracks):
        out_frames = []
        for frame_id, frame in enumerate(frames):
            copy = frame.copy()

            player_dictionary = tracks["player"][frame_id]
            referee_dictionary = tracks["referee"][frame_id]
            ball_dictionary = tracks["football"][frame_id]

            for track_id, player in player_dictionary.items():
                color = player.get("team_color", (255,255,255))
                self.draw_ellipse(frame, player["bbox"], color, track_id=track_id)

            for track_id, referee in referee_dictionary.items():
                self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))
            
            for track_id, ball in ball_dictionary.items():
                self.draw_triangle(frame, ball['bbox'], (0,255,0))
            out_frames.append(frame)
        return out_frames