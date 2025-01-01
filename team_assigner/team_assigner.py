from sklearn.cluster import KMeans


def get_clustering_model(image, fitting_or_not, player_colors=None):
    # Reshape the image into 2d array
    if not fitting_or_not:
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(image_2d)
    else:
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(player_colors)
    return kmeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.kmeans = None
        self.player_team_dict = {}

    def get_player_color(self, frame, bounding_box):
        image = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]
        top_half = image[:int(image.shape[0] / 2)]
        kmeans = get_clustering_model(top_half, False)
        labels = kmeans.labels_
        # reshape the labels to the original image size
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])
        corner_clusters = [clustered_image[0][0], clustered_image[0][-1], clustered_image[-1][0],
                           clustered_image[-1][-1]]
        count_0 = 0
        count_1 = 0
        for element in corner_clusters:
            if element == 0:
                count_0 += 1
            else:
                count_1 += 1
        if count_0 > count_1:
            non_player_cluster = 0
            player_cluster = 1
        else:
            non_player_cluster = 1
            player_cluster = 0

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bounding_box = player_detection["bounding_box"]
            player_color = self.get_player_color(frame, bounding_box)
            player_colors.append(player_color)

        kmeans = get_clustering_model(None, True, player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bounding_box, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_color = self.get_player_color(frame, player_bounding_box)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
        self.player_team_dict[player_id] = team_id
        return team_id
