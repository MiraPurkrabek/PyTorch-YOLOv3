import json

# Read it back
with open("via_export_json.json") as json_file:
    data = json_file.read()

# decoding the JSON to dictionary
d = json.loads(data)

# print(type(d))
# print(d["frame00000.jpg601425"]["regions"][0])
image_width = 1920
image_heigh = 1080


frame_num = 0
image_list = ''
for tmp in d.items():
    frame = tmp[1]
    img_name = frame["filename"]
    filename = "export/{}.txt".format(img_name[:-4])
    detections = ''
    for bbox in frame["regions"]:
        bb_class = bbox["region_attributes"]["type"]
        x = bbox["shape_attributes"]["x"]
        y = bbox["shape_attributes"]["y"]
        width = bbox["shape_attributes"]["width"]
        height = bbox["shape_attributes"]["height"]
        x_center = x+width/2
        y_center = y+width/2

        # Decide class in numbers
        bb_class_num = 3 # Default is ref
        if bb_class == "player_A":
            bb_class_num = 0 
        elif bb_class == "player_B":
            bb_class_num = 1
        elif bb_class == "goalie":
            bb_class_num = 2
        detections += "{} {:4f} {:4f} {:4f} {:4f}\n".format(bb_class_num, x_center/image_width, y_center/image_heigh, width/image_width, height/image_heigh)
        
    # If any detections, save into file
    if len(detections) > 0:
        image_list += filename + "\n"
        with open(filename, "w") as save_file:
            save_file.write(detections)
        # print('{}\t{}, {}, {} ,{}, {}'.format(img_name[:-4], bb_class, x, y, width, height))
    frame_num += 1

with open("images_to_train", "w") as train:
    train.write(image_list)
