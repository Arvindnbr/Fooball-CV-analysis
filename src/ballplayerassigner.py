import sys
sys.path.append("../")

from utils.bbox_utils import get_bbox_centre, get_distance


class BallerAssigner:
    def __init__(self):
        self.max_distaance = 50

    def ball_to_player_assign(self, players, ball_bbox):

        ball = get_bbox_centre(ball_bbox)

        min_distance = 9999
        assigned_player = -1

        for playerid, player in players.items():
            bbox = player["bbox"]

            left_distance = get_distance((bbox[0],bbox[-1]),ball)
            right_distance = get_distance((bbox[2],bbox[-1]),ball)
            distance = min(left_distance,right_distance)
            
            if distance < self.max_distaance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = playerid

        return assigned_player