import os
import xml.etree.ElementTree as ET

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file_path, output_txt_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(output_txt_path, 'w') as out_file:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} " + " ".join([str(a) for a in bb]) + '\n')

def convert_annotations_from_directory(xml_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    xml_files = [f for f in os.listdir(xml_directory) if f.endswith('.xml')]
    for xml_file in xml_files:
        xml_file_path = os.path.join(xml_directory, xml_file)
        output_txt_path = os.path.join(output_directory, os.path.splitext(xml_file)[0] + '.txt')
        convert_annotation(xml_file_path, output_txt_path)

classes = ["fire", "smoke"]  


convert_annotations_from_directory('VOC2020/Annotations', 'VOC2020/Annotations/labels')
