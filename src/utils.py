def filter_classes(results, target_classes):
    filtered_boxes = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = results[0].names[cls]

        if label in target_classes:
            filtered_boxes.append((box, label))

    return filtered_boxes


def count_objects(filtered_boxes):
    counts = {}

    for _, label in filtered_boxes:
        counts[label] = counts.get(label, 0) + 1

    return counts