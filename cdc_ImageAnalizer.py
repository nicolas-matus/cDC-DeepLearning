import time
import os
import re
import cdc_functions as cdc #módulo con las funciones de análisis de imágenes, debería estar en la misma carpeta que este script

# Valores de configuración por defecto (backup si no hay archivo config.txt)
DEFAULT_CONFIG = {
    "classModelDir": r".\models\classification\ResNet101_classification.keras",
    "bboxModelDir": r".\models\BBox\Mobilenet_BBox\saved_model",
    "channelModelDir": r".\models\segmentation\channelModel.keras",
    "segmentationModelDir": r".\models\segmentation\cellUnet.keras",
    "imDir": r".\images",
    "outputDir": r".\Results",
    "startImgNum": 0,
    "endImgNum": 0,
    "batchSize": 64,
    "classificationThreshold": 0.4,
    "cellSegmentationThreshold": 0.55,
    "channelSegmentationThreshold": 0.15,
    "BBoxThreshold": 0.1,
}

def load_config(config_path):
    """ Carga la configuración desde un archivo de texto.
        Si el archivo no existe, usa los valores por defecto."""
    config = DEFAULT_CONFIG.copy()  # Carga los valores por defecto
    
    if not os.path.exists(config_path):
        print("No config file found, using default paths.")
        return config

    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignora líneas vacías y comentarios que empiezan con #
            if not line or line.startswith('#'):
                continue
            #Se asume que las líneas tienen el formato "key: value"
            # Divide en key: value
            if ':' in line:
                key, value = line.split(':', 1)  
                key = key.strip()
                value = value.strip()
                
                # Convierte índices y batch size a int
                if key in ["startImgNum", "endImgNum", "batchSize"]:
                    try:
                        value = int(value)
                    except ValueError:
                        print(f"Warning: Invalid number for {key}. Using default.")
                        value = DEFAULT_CONFIG[key]
                # Convierte puntajes de corte a float
                if key in ["classificationThreshold", "cellSegmentationThreshold", "channelSegmentationThreshold", "BBoxThreshold"]:
                    try:
                        value = float(value)
                    except ValueError:
                        print(f"Warning: Invalid number for {key}. Using default.")
                        value = DEFAULT_CONFIG[key]
                
                config[key] = value

    return config


def extract_last_number(filename):
    """ Permite encontrar el último número en el nombre del archivo.
        Se asume que el formato del nombre del archivo es algo como "imagen_123.tiff" u otra extensión"""
    match = re.search(r'_(\d+)\.\w+$', filename)
    return int(match.group(1)) if match else -1

# Carga la configuración
config = load_config(r".\config.txt")

# Asigna las variables 
classModelDir = config["classModelDir"]
bboxModelDir = config["bboxModelDir"]
channelModelDir = config["channelModelDir"]
segmentationModelDir = config["segmentationModelDir"]
imDir = config["imDir"]
outputDir = config["outputDir"]
startImgNum = config["startImgNum"]
endImgNum = config["endImgNum"]
batchSize = config["batchSize"]
classificationThreshold = config["classificationThreshold"]
cellSegmentationThreshold = config["cellSegmentationThreshold"]
channelSegmentationThreshold = config["channelSegmentationThreshold"]
BBoxThreshold = config["BBoxThreshold"]

# Define el rango de imágenes a trabajar según startImgNum y endImgNum
# Si startImgNum y endImgNum son 0, se procesan todas las imágenes
slice_obj = slice(None) if startImgNum == 0 and endImgNum == 0 else slice(startImgNum, endImgNum)
os.makedirs(outputDir,exist_ok=True)
image_files = [os.path.join(imDir, f) for f in os.listdir(imDir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]


# Ordena los archivos de imagen por el último número en el nombre del archivo
# Si no se ordenaran, el sistema podría considerar que la imagen 10 es menor que la 2, por ejemplo.
image_files.sort(key=extract_last_number)
image_files = image_files[slice_obj]

totalImages = len(image_files)

if image_files == 0:
    raise ValueError("No images found in the specified directory.")

start_time = time.time()
# Localiza las cajas delimitadoras en las constricciones, extrae el recorte,
# determina si el canal está ocupado y cuenta el número de canales bloqueados.
results = cdc.bboxInference(image_files, bboxModelDir, threshold=BBoxThreshold)

# Se determina qué canales ocupados contienen células.
channelData = cdc.classification(classModelDir, results, batchSize= batchSize, threshold=classificationThreshold)
del results

# Agrupa las imágenes de células según segmentos continuos.
experimentsList = cdc.getExperimentList(channelData)
del channelData

# Calcula las propiedades geométricas de las células, y su avance en el tiempo.
expOutput= cdc.cellSegmentation(segmentationModelDir, experimentsList, displacementThreshold=1, batchSize = batchSize, threhold=cellSegmentationThreshold)

# Segmenta los canales y guarda los resultados en un directorio de salida.
# Guarda un archivo con los metadatos de cada experimento.
metadataList = cdc.segmentChannelAndSaveReults(channelModelDir, outputDir, expOutput, startIndex=1)
metadataFile = os.path.join(outputDir, 'metadata.txt')
with open(metadataFile, 'w') as f:
    f.write("expNum Rc wc BlCH firstFrame lastFrame\n")
    for metadata in metadataList:
        f.write(f"{metadata['expNum']} {metadata['Rc']} {metadata['wc']} {metadata['occupied']} {metadata['firstFrame']} {metadata['lastFrame']}\n")



# Se indica el tiempo total de ejecución del script.
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
