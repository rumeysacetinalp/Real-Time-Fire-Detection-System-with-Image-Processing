import os
import xml.etree.ElementTree as ET

# Kaynak ve hedef dizinleri belirleyin
source_dir = 'C:/Users/LENOVO/Desktop/annotations'  # XML dosyalarının bulunduğu klasör
target_dir = 'C:/Users/LENOVO/Desktop/labels'  # .txt dosyalarının kaydedileceği klasör

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# XML'den YOLO formatına dönüştürme fonksiyonu
def convert_xml_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    file_name = os.path.splitext(root.find('filename').text)[0]
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    
    txt_file_path = os.path.join(target_dir, f'{file_name}.txt')
    with open(txt_file_path, 'w') as txt_file:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name == 'smoke':
                class_id = 1  # Smoke sınıfı için sınıf etiketi 1
            else:
                continue  # Diğer sınıfları atla
            
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            
            # YOLO formatına dönüştür
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / float(img_width)
            height = (ymax - ymin) / float(img_height)
            
            txt_file.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

# Tüm XML dosyalarını işleyin
for xml_file in os.listdir(source_dir):
    if xml_file.endswith('.xml'):
        convert_xml_to_yolo(os.path.join(source_dir, xml_file))

print("Dönüştürme tamamlandı!")
