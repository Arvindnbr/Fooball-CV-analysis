from utils import save_vid, read_vid
from trackers import Tracker
import cv2
import numpy as np
from src import TeamAssigner, BallerAssigner, CameraMovement

def main():
    #read video
    video_frames = read_vid("videos/bundesliga.mp4")

    #tracker
    tracker = Tracker("training/v5l.pt")
    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_pkl= True,
                                       pkl_path= "/home/arvind/Python/Fooball-CV-analysis/videos/pkl/tracks1.pkl")
    
    #add cameramovements to the tracks
    tracker.add_position_to_tracks(tracks)
    
    cameramovement = CameraMovement(video_frames[0])
    cameramovement_perframe = cameramovement.get_camera_movement(video_frames, 
                                                                 read_from_pkl=True,
                                                                 pkl_path="videos/pkl/camera_movement1.pkl")
    cameramovement.adjusted_positions_to_track(tracks,cameramovement_perframe)

    #interpolate ball movements
    tracks["football"] = tracker.interpolate_ball_tracks(tracks["football"])
    
    #assign teams
    assigner = TeamAssigner()
    #print(tracks["player"][0])
    assigner.team_color(video_frames[0],
                    tracks["player"][0])
    
    for frame_no, player_track in enumerate(tracks["player"]):
        for player_id , track in player_track.items():
            team = assigner.get_player_team(video_frames[frame_no],
                                            track['bbox'],
                                            player_id)
            tracks["player"][frame_no][player_id]['team'] = team
            tracks["player"][frame_no][player_id]['team_color'] = assigner.team_colors[team]

    #ball assigner
    ballassigner = BallerAssigner()
    team_ball_contol = [1]
    for frame_num, player_track in enumerate(tracks['player']):
        ball_bbox = tracks['football'][frame_num][1]['bbox']
        assignd_player = ballassigner.ball_to_player_assign(player_track,ball_bbox)

        if assignd_player != -1:
            tracks['player'][frame_num][assignd_player]['has_ball'] = True
            team_ball_contol.append(tracks["player"][frame_num][assignd_player]['team']) 
        else:
            team_ball_contol.append(team_ball_contol[-1])  

    team_ball_contol = np.array(team_ball_contol)     
    
    
    # for track_id, player in tracks['player'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    #     cv2.imwrite("/home/arvind/Python/Fooball-CV-analysis/videos/output/cropped.jpg",cropped_image)
    #     break
    op_frame = tracker.plot_annotations(video_frames, tracks, team_ball_contol)

    #plot camera movement 
    op_frame = cameramovement.plot_camera_movement(video_frames, cameramovement_perframe) 


    save_vid(op_frame, "/home/arvind/Python/Fooball-CV-analysis/videos/output/op1.avi")
    





if __name__ == "__main__":
    main()