import xml.etree.ElementTree as ET
import os

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    
    # We'll use "blood" as class name since your <name> tag contains filenames
    class_name = "blood"
    class_id = 0
    
    txt_file = xml_file.replace(".xml", ".txt")
    with open(txt_file, "w") as out:
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.find("xmin").text)
            xmax = float(bndbox.find("xmax").text)
            ymin = float(bndbox.find("ymin").text)
            ymax = float(bndbox.find("ymax").text)
            bb = convert((w, h), (xmin, xmax, ymin, ymax))
            out.write(f"{class_id} {' '.join([str(a) for a in bb])}\n")
    print(f"✅ Converted: {xml_file} → {txt_file}")

for file in os.listdir("."):
    if file.endswith(".xml"):
        convert_annotation(file)

print("🎯 All XML files converted successfully!")
