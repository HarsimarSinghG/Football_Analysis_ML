class PlayerBallAssigner:
    def __init__(self, max_player_ball_distance=70):
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(self, players, ball_bounding_box):
        ball_center = self._get_center_of_bbox(ball_bounding_box)
        closest_distance = float('inf')  # Initialize with a very large number
        player_assigned = None

        for player_id, player_data in players.items():
            player_bbox = player_data["bounding_box"]
            distances = self._calculate_distances(player_bbox, ball_center)

            # Choose the smaller distance between left and right side of the player bounding box
            min_distance = min(distances)

            # Check if the player is within the max distance threshold
            if min_distance < self.max_player_ball_distance and min_distance < closest_distance:
                closest_distance = min_distance
                player_assigned = player_id

        return player_assigned

    def _get_center_of_bbox(self, bounding_box):
        """
        Helper method to calculate the center of a bounding box
        """
        x_center = (bounding_box[0] + bounding_box[2]) / 2
        y_center = (bounding_box[1] + bounding_box[3]) / 2
        return (x_center, y_center)

    def _calculate_distances(self, player_bbox, ball_position):
        """
        Calculate the distances from both the left and right corners of the player bounding box to the ball position
        """
        left_corner = (player_bbox[0], player_bbox[-1])
        right_corner = (player_bbox[2], player_bbox[-1])
        return [self._measure_distance(left_corner, ball_position),
                self._measure_distance(right_corner, ball_position)]

    def _measure_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points
        """
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
