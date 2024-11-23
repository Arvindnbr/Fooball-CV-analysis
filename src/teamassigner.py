import numpy as np
from sklearn.cluster import KMeans




class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}      #player_id : team1 or 2
    # get each players from the frame run k means on them and  find the color
    # forr that assign 2 colors for the team
    # predict weather the player belong to that team
    def get_clustering_model(self, frame):
        image_2d = frame.reshape(-1,3)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)
        return kmeans

    
    def get_player_color(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        cropped = img[:int(img.shape[0]),:]
        kmeans = self.get_clustering_model(cropped)

        #get cluser labels
        labels = kmeans.labels_

        clustered_img = labels.reshape(cropped.shape[0], cropped.shape[1])

        #get player cluster
        corner = [clustered_img[0,0],clustered_img[0,-1],clustered_img[-1,0],clustered_img[-1,-1]]
        bg_cluster = max(set(corner), key=corner.count)
        player_cluster = 1 if bg_cluster == 0 else 0

        #get cluster center
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color





    def team_color(self, frame, player_detection):

        playr_colors = []
        for _ , player in player_detection.items():
            bbox = player["bbox"]
            player_color = self.get_player_color(frame,bbox)
            playr_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(playr_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id +=1

        self.player_team_dict[player_id] = team_id

        return team_id
