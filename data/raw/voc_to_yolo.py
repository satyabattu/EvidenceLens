import os
import xml.etree.ElementTree as ET

# 📁 input and output folders
input_dir = "/mnt/c/Users/satya/OneDrive/Desktop/Evidencelens/data/raw/"
output_dir = os.path.join(input_dir, "yolo_labels")
os.makedirs(output_dir, exist_ok=True)

# 🏷️ your class names (order matters)
classes = ["blood", "gun", "knife"]

for xml_file in os.listdir(input_dir):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(input_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        continue
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # normalize for YOLO
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    # save as .txt
    txt_file = os.path.join(output_dir, xml_file.replace(".xml", ".txt"))
    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_lines))
