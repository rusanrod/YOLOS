import cv2
import numpy as np
import json
import os
from matplotlib import pyplot as plt

def process_masks_with_annotations(image_dir, mask_dir, output_dir, json_path, kernel_size=3, category_id=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotations_list = []
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for idx, image_name in enumerate(os.listdir(image_dir), start=1):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)

        # Leer imagen y máscara
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Error al leer {image_path} o {mask_path}")
            continue

        # Erosión y dilatación
        eroded = cv2.erode(mask, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        # Encontrar contornos
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Generar anotaciones
        image_annotations = {
            "image_id": idx,
            "image_name": image_name,
            "annotations": []
        }


        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min, y_min = x, y
            x_max, y_max = x + w, y + h
            area = w * h
            bbox = [x_min, y_min, x_max, y_max]  # Formato [x_min, y_min, x_max, y_max]
            annotation = {
                "bbox": bbox,
                "category_id": category_id,
                "area": area
            }
            image_annotations["annotations"].append(annotation)

            # Dibujar bounding box en la imagen para visualización
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        annotations_list.append(image_annotations)

        # Guardar imagen con bounding boxes
        output_image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_image_path, image)

    # Guardar JSON
    with open(json_path, 'w') as f:
        json.dump(annotations_list, f, indent=4)

    print(f"Procesamiento terminado. Etiquetas guardadas en {json_path}.")

    # Mostrar preview de algunas imágenes
    for preview_image in os.listdir(output_dir)[:5]:
        preview_path = os.path.join(output_dir, preview_image)
        preview_img = cv2.imread(preview_path)
        plt.imshow(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB))
        plt.title(preview_image)
        plt.axis("off")
        plt.show()

# Directorios de ejemplo
image_dir = "./coronacases/frames"
mask_dir = "./coronacases/masks"
output_dir = "./coronacases/outputs"
json_path = "./coronacases/labels.json"

process_masks_with_annotations(image_dir, mask_dir, output_dir, json_path)
