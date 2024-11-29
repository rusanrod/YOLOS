import os
import pydicom
from PIL import Image
import numpy as np

def convert_dcm_folder(input_folder, output_folder):
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Recorrer todos los archivos en la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(".dcm"):  # Filtrar solo archivos .dcm
            dcm_path = os.path.join(input_folder, filename)
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)
            
            try:
                # Leer y convertir el archivo DICOM
                dicom = pydicom.dcmread(dcm_path)
                pixel_array = dicom.pixel_array
                
                # Normalizar los valores a rango 0-255
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
                pixel_array = pixel_array.astype(np.uint8)
                
                # Guardar como JPG
                img = Image.fromarray(pixel_array)
                img.save(jpg_path)
                print(f"Convertido: {filename} -> {jpg_filename}")
            except Exception as e:
                print(f"Error al convertir {filename}: {e}")
# Configurar carpetas
input_folder = "./neumonia/CT_scans/PAT001"
output_folder = "./neumonia/outputs"

# Convertir toda la carpeta
print("hola")
convert_dcm_folder(input_folder, output_folder)