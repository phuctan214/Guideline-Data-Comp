from PIL import Image
import ntpath
import os
import glob

# convert yolo to pascal voc format
def yolo_convert_pascal_voc(box_yolo, h, w, label):
    a = int((box_yolo[0] * 2 * w - box_yolo[2] * w) / 2)
    c = int((box_yolo[0] * 2 * w + box_yolo[2] * w) / 2)
    b = int((box_yolo[1] * 2 * h - box_yolo[3] * h) / 2)
    d = int((box_yolo[1] * 2 * h + box_yolo[3] * h) / 2)
    return [label, a, b, c, d]


# convert pascal voc to yolo format
def pascal_voc_convert_yolo(list_box, labels, w, h):
    new_list_box = []
    new_boxes = list_box.tolist()
    labels = labels.tolist()
    for i, box in enumerate(list_box):
        a, b, c, d = new_boxes[i]
        new_boxes[i][0] = (a + c) / 2 / w
        new_boxes[i][1] = (b + d) / 2 / h
        new_boxes[i][2] = (c - a) / w
        new_boxes[i][3] = (d - b) / h
        new_boxes[i].insert(0, int(labels[i]))
    return new_boxes


# merge all bouding box
def yolo_format_to_str(list_box):
    for i in range(len(list_box)):
        list_box[i] = [str(e) for e in list_box[i]]
        list_box[i] = " ".join(list_box[i])

    return "\n".join(list_box)


def parse_annot(image_path, folder_annotation):
    image = Image.open(image_path, mode="r")
    image = image.convert("RGB")
    w, h = image.size

    # anno_path = image_path.replace("images", "labels")
    # anno_path = anno_path.replace("jpg", "txt")

    name_file = ntpath.basename(image_path).split(".")[0]
    anno_path = folder_annotation + f"{name_file}.txt"

    with open(anno_path, 'r') as f:
        anno_data = f.read()
    f.close()
    anno_data = anno_data.strip()
    if anno_data == "":
        return []
    else:
        boxes = list()
        labels = list()

        data_test = anno_data.split("\n")
        data_test = [i for i in data_test if i != ""]
        data_test = [i.split(" ") for i in data_test]
        for i, _ in enumerate(data_test):
            tmp = data_test[i]
            tmp = [float(e) for e in tmp]
            box_descrip = tmp[1:]
            new_box = yolo_convert_pascal_voc(box_yolo=box_descrip, h=h, w=w, label=int(tmp[0]))
            boxes.append(new_box[1:])
            labels.append(new_box[0])

        return {"boxes": boxes, "labels": labels}

def find_and_remove_empty_file(folder_name:str):
    for file_path in glob.glob(folder_name + "/*.*"):
        if os.path.getsize(file_path) == 0:
            os.remove(file_path)