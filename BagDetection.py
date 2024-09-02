import numpy as np


def find_nearest_index(array, value):
    distance = np.linalg.norm(np.array(array) - np.array(value))
    distances = []
    for point in array:
        distance = np.linalg.norm(np.array(point) - np.array(value))
        distances.append(distance)
    idx = np.argmin(np.array(distances))
    return idx


def compute_iou(boxA, boxB):
    print(boxA, boxB)
    x1, y1, x2, y2 = boxA
    x3, y3, x4, y4 = boxB
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    bag_area = abs(x2 - x1) * abs(y2 - y1)
    print(x_inter1, y_inter1, x_inter2, y_inter2)
    print(x3, y3, x4, y4)
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    print(area_inter)
    print(bag_area)
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / bag_area

    print(iou)
    return iou


def get_related_person(tracks, object_cords):

    for object in tracks:
        print(object)
        cls = object[6]

        if cls == 0:
            if compute_iou(object_cords, object[:4]) > 0.1:
                return object[4]
    return None


def get_person_trackid(tracks):
    person_trackids = []
    for object in tracks:
        cls = object[6]
        if cls == 0:
            person_trackids.append(object[4])
    return person_trackids


def check_abandon(tracks, thresh, list_of_bags):
    for object in tracks:
        cls = object[6]
        if cls == 26 or cls == 24 or cls == 28:
            x1, y1, x2, y2 = object[0], object[1], object[2], object[3]
            center = [int((x2 + x1) / 2), int((y2 + y1) / 2)]

            if len(list_of_bags) > 0:
                first_column = [sublist[0] for sublist in list_of_bags]

                nearest_index = find_nearest_index(first_column, center)
                nearest_point = list_of_bags[nearest_index]
                # Compare distance (L2 norm) between center and nearest_point
                if np.linalg.norm(np.array(center) - np.array(nearest_point[0])) < 10:
                    # Increment the count at the nearest point

                    list_of_bags[nearest_index][1] += 1

                else:
                    # Append a new point [center, 1] to list_of_bags
                    related_person_id = get_related_person(tracks, object[:4])
                    list_of_bags.append(
                        [center, 1, related_person_id, object[:4]])
            else:
                # Append the first point [center, 1] to list_of_bags
                related_person_id = get_related_person(tracks, object[:4])
                list_of_bags.append([center, 1, related_person_id, object[:4]])
        # Check if any bag count exceeds the threshold

    for bag in list_of_bags:
        if len(bag) > 1 and bag[1] >= thresh and (bag[2] is None or bag[2] not in get_person_trackid(tracks)):

            print(f"bag:- {bag}")
            return (True, bag[0], bag[3])

    # Return False if no bag count exceeds the threshold
    return (False, None, None)

# Test cases


if __name__ == "__main__":
    # list_of_bags = []
    # tracks = [
    #     [10, 20, 30, 40, 1, 1, 24],
    #     [50, 20, 60, 40, 1, 1, 26],
    #     [20, 20, 100, 50, 3, 1, 0],
    # ]
    # print(check_abandon(tracks, 3, list_of_bags))
    # print(list_of_bags)

    # tracks = [
    #     [200, 20, 100, 40, 1, 1, 24],
    #     [48, 22, 58, 37, 1, 1, 26],
    #     [20, 20, 100, 50, 3, 1, 0],
    # ]
    # print(check_abandon(tracks, 3, list_of_bags))
    # print(list_of_bags)

    # tracks = [
    #     [200, 20, 100, 40, 1, 1, 24],
    #     [50, 20, 60, 40, 2, 1, 26],
    #     [20, 20, 100, 50, 3, 1, 0],
    # ]
    # print(check_abandon(tracks, 3, list_of_bags))
    # print(list_of_bags)
    # print(compute_iou([424, 570, 595, 701], [381, 574, 830, 852]))
    # print(compute_iou([50, 100, 200, 300],[80, 120, 220, 310]))


# 424         570         595         701           5     0.31203          24           3]
# [        997          23        1027          89

# 381         574         830         852

# 430         572         595         692           5     0.33602          24           2]
# [        379         571         830         852

    bag = [430, 572, 595, 300, 692, 0.48689, 24, 3]
    person = [ 379, 571, 830, 852, 1, 0.86468, 0, 0]
    print(get_related_person([person], bag[:4]))
