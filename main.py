from utils import save_vid, read_vid
from trackers import Tracker
import cv2
from src import TeamAssigner

def main():
    #read video
    video_frames = read_vid("videos/op3.mp4")

    #tracker
    tracker = Tracker("training/v5l.pt")
    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_pkl= True,
                                       pkl_path= "/home/arvind/Python/Fooball-CV-analysis/videos/pkl/tracks.pkl")
    
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
            
    
    # for track_id, player in tracks['player'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    #     cv2.imwrite("/home/arvind/Python/Fooball-CV-analysis/videos/output/cropped.jpg",cropped_image)
    #     break
    op_frame = tracker.plot_annotations(video_frames, tracks)


    save_vid(op_frame, "/home/arvind/Python/Fooball-CV-analysis/videos/output/op1.avi")
    





if __name__ == "__main__":
    main()