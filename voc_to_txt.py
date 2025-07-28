import os
import xml.etree.ElementTree as ET

# Class names
CLASSES = [
    'car', 'suv', 'van', 'bus', 'freight_car', 'truck',
    'tank_truck', 'trailer', 'crane', 'excavator', 'motorcycle'
]

# VOC to YOLO format conversion
def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

# Convert one XML annotation to txt
def convert_annotation(xml_file, label_dir, txt_dir):
    tree = ET.parse(os.path.join(label_dir, xml_file))
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    txt_path = os.path.join(txt_dir, xml_file.replace('.xml', '.txt'))
    with open(txt_path, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in CLASSES:
                print(f"Unknown class: {cls}")
                continue
            cls_id = CLASSES.index(cls)
            xmlbox = obj.find('bndbox')
            b = (
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text)
            )
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")

if __name__ == '__main__':
    # Input XML label folder path
    label_dir = os.path.expanduser(input('Enter the absolute path to the XML label folder: '))
    parent_dir = os.path.dirname(label_dir)
    txt_dir = os.path.join(parent_dir, 'labels_txt')
    os.makedirs(txt_dir, exist_ok=True)
    for xml_file in os.listdir(label_dir):
        if xml_file.endswith('.xml'):
            print(f"Processing: {xml_file}")
            convert_annotation(xml_file, label_dir, txt_dir)