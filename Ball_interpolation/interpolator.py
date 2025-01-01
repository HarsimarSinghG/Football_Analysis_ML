import pandas as pd


class Interpolator:
    def __init__(self):
        pass

    def interpolate_ball_positions(self, ball_positions):
        # Extracting bounding box data from ball positions and converting to DataFrame
        processed_positions = []
        for pos in ball_positions:
            bounding_box = pos.get(1, {}).get("bounding_box", [])
            processed_positions.append(bounding_box)

        ball_df = pd.DataFrame(processed_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolating missing values (NaN) in the DataFrame
        ball_df = ball_df.apply(pd.to_numeric, errors='coerce')  # Ensure numerical conversion
        ball_df = ball_df.interpolate(method='linear')  # Interpolating missing data
        # Backfilling will be necessary if there are NaNs at the beginning of the DataFrame
        ball_df = ball_df.bfill()  # Backfill any remaining NaNs

        # Converting the DataFrame back to the original dictionary format
        interpolated_positions = []
        for _, row in ball_df.iterrows():
            interpolated_positions.append({1: {"bounding_box": row.tolist()}})

        return interpolated_positions

