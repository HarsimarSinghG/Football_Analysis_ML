from sklearn.cluster import KMeans


# Function to perform k-means clustering on the image or player colors
def get_clustering_model(image, fitting_or_not, player_colors=None):
    # Reshape image into a 2D array if clustering on the image
    if not fitting_or_not:
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(image_2d)
    else:
        # Perform clustering on the provided player colors
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(player_colors)
    return kmeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}  # Dictionary to store team colors
        self.kmeans = None  # KMeans model for clustering player colors
        self.player_team_dict = {}  # Mapping of player IDs to team IDs

    # Extract dominant color of a player from a given frame and bounding box
    def get_player_color(self, frame, bounding_box):
        # Crop the image to the player’s bounding box
        image = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]
        top_half = image[:int(image.shape[0] / 2)]  # Consider top half for clustering
        kmeans = get_clustering_model(top_half, False)  # Cluster top half
        labels = kmeans.labels_  # Get cluster labels

        # Reshape labels to match the image dimensions
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])

        # Get corner pixels from the clustered image
        corner_clusters = [clustered_image[0][0], clustered_image[0][-1], clustered_image[-1][0],
                           clustered_image[-1][-1]]
        count_0 = 0
        count_1 = 0
        # Count occurrences of each cluster in the corners
        for element in corner_clusters:
            if element == 0:
                count_0 += 1
            else:
                count_1 += 1
        # Assign clusters based on the majority in the corners
        if count_0 > count_1:
            non_player_cluster = 0
            player_cluster = 1
        else:
            non_player_cluster = 1
            player_cluster = 0

        # Return the color of the player cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    # Assign players to teams based on their colors
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bounding_box = player_detection["bounding_box"]
            # Get the player's dominant color
            player_color = self.get_player_color(frame, bounding_box)
            player_colors.append(player_color)

        # Perform clustering on player colors to define team colors
        kmeans = get_clustering_model(None, True, player_colors)
        self.kmeans = kmeans
        # Assign cluster centers as team colors
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    # Get the team assignment for a player
    def get_player_team(self, frame, player_bounding_box, player_id):
        # Check if the player’s team is already assigned
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Get the player’s color and predict the team
        player_color = self.get_player_color(frame, player_bounding_box)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1  # Predict team
        self.player_team_dict[player_id] = team_id  # Store the team assignment
        return team_id
