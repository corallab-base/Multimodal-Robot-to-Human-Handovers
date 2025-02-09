import os
import json

def get_bounding_boxes(class_name, filename, base_dir="gaze_extra"):
    """
    Reads bounding box information for a given image file from a class-specific label file.

    Parameters:
    - class_name (str): The category/class name (e.g., 'marble', 'clamp', 'flashlight').
    - filename (str): The specific image filename (e.g., 'image_004.png').
    - base_dir (str): The base directory where the label files are stored.

    Returns:
    - List of tuples [(x1, x2, y1, y2), ...] representing bounding boxes.
    - The "left" bounding box always comes first, followed by the "right" bounding box.
    - Returns an empty list if the image is not found in the label file.
    """

    label_file_path = os.path.join(base_dir, f"labels_{class_name}.txt")

    if not os.path.exists(label_file_path):
        print(f"Label file {label_file_path} not found.")
        return []

    try:
        # Load the entire JSON file
        with open(label_file_path, "r") as file:
            data = json.load(file)  # Assumes valid JSON list in the file
        
        # Find the image entry
        for entry in data:
            if entry.get("key") == filename:
                left_boxes = []
                right_boxes = []

                for box in entry.get("boxes", []):
                    x1 = float(box["x"])
                    y1 = float(box["y"])
                    x2 = x1 + float(box["width"])
                    y2 = y1 + float(box["height"])

                    if box["label"].lower() == "left":
                        left_boxes.append((x1, x2, y1, y2))
                    elif box["label"].lower() == "right":
                        right_boxes.append((x1, x2, y1, y2))

                # Return left bounding boxes first, then right
                return left_boxes + right_boxes

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {label_file_path}: {e}")

    print(f"No bounding box data found for {filename} in {class_name}.")
    return []


if __name__ == '__main__':
    bb = get_bounding_boxes('flashlight', 'image_000.png')
    print(bb)