import numpy as np

def calculate_iou(pred, label):
    # Ensure the predictions and labels are binary masks
    pred = (pred > 0).astype(np.uint8)
    label = (label > 0).astype(np.uint8)

    # Calculate intersection and union
    intersection = np.logical_and(pred, label).sum()
    union = np.logical_or(pred, label).sum()

    # Avoid division by zero
    if union == 0:
        return 0.0

    iou = intersection / union
    return iou

def evaluate_lane_detection(predictions, labels, iou_threshold=0.5):
    results = []
    for pred, label in zip(predictions, labels):
        iou = calculate_iou(pred, label)
        is_same = iou > iou_threshold
        results.append({
            'iou': iou,
            'is_same': is_same
        })
    return results

# Example usage:
# predictions = [np.array(...), np.array(...)]  # List of predicted lane masks
# labels = [np.array(...), np.array(...)]  # List of ground truth lane masks
# evaluation_results = evaluate_lane_detection(predictions, labels)