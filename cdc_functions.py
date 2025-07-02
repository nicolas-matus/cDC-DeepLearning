# Programa desarrollado por Nicolás Ignacio Matus Millar (2025)
# Parte de la tesis de ingeniería civil mecánica de la Universidad de Santiago de Chile.

# Desarrollo de una técnica de análisis automático de imágenes para la
# caracterización mecánica celular en dispositivos tipo cDC mediante
# Deep Learning


# Funciones para el análisis de imágenes de células en canales de un dispositivo de microfluidos 
# de citometría de deformabilidad basado en constricción (CDC).

# Este archivo está pensado para ser importado como un módulo en otros scripts de Python.

# Se recomienda trabajar con Python 3.9.18 y tensorflow 2.10.1
# Estas funciones fueron desarrolladas en Windows 11 con CUDA 11.2 y CUDNN 9.7


import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import re

import time


def extract_last_number(filename):
    match = re.search(r'_(\d+)\.\w+$', filename)
    return int(match.group(1)) if match else -1

def load_image_into_numpy_array(image_path):
    """Entradas: Ruta de la imagen
    Salidas: Imagen en formato de array numpy, con los colores en formato RGB

    Convierte la ruta de la imagen a un array de numpy que puede ser utilizado por la red neuronal
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np = np.expand_dims(image, axis=0)
    return image_np

def load_batch_into_numpy_array(batchPaths):
    """Entradas: Lista con las rutas de las imágenes del lote
    Salidas: Array de numpy con las imágenes del lote, con los colores en formato RGB

    Convierte la ruta de la imagen a un array de numpy que puede ser utilizado por la red neuronal
    """
    images = []
    for image_path in batchPaths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    batch_np = np.stack(images, axis=0)
    return batch_np

def upscale_image(image_np, target_size):
    """Cambia la resolución de una imagen a un tamaño objetivo"""
    return cv2.resize(image_np, target_size, interpolation=cv2.INTER_NEAREST)

def find_constrict_edge(mask, s=5):
    """Entradas: Máscara de un canal, opcionalmente se puede elegir el número de pixeles a ignorar en los extremos
    Salidas: Posición del borde de la constricción del canal, y las coordenadas de los puntos del borde

    Encuentra el borde de la constricción de un canal, buscando la posición en la que la distancia vertical es mínima"""
    mask = (mask > 0.5).astype(np.uint8)
    mask[:, :s] = 1 #Permite ignorar los primeros y últimos s pixeles de la máscara
    mask[:, -s:] = 1
    # Calcula el largo de cada línea vertical del canal, y considera que el borde izquierdo de la constricción 
    # corresponde a la posición en la que la distancia vertical es mínima, escogiendo la que esté 
    # más a la izquierda si hay varias posiciones con la misma distancia mínima.
    start = np.argmax(mask, axis=0)
    stop = mask.shape[0] - np.argmax(np.flip(mask, axis=0), axis=0) - 1
    lenght = stop - start + 1
    middle_index = np.argmin(lenght)

    return middle_index, start, stop


def find_constrict_center(mask, box=None, s=5):
    """Entradas: Máscara de un canal y la caja de la detección, opcionalmente se puede elegir el número de pixeles a ignorar
    Salidas: Puntos del centro de la constricción del canal, reescalados a las dimensiones de la imagen original

    Encuentra el centro de la constricción de un canal, calculando el promedio de las posiciones de los bordes
    """

    #Calcula el borde de la constricción del canal por la izquierda y derecha
    xi, start, stop = find_constrict_edge( mask, s)  # Encuentra la posición desde la izquierda
    xd, _, _ = find_constrict_edge( np.flip(mask, axis=1), s)  # Encuentra la posición desde la derecha
    xd = range( mask.shape[0])[ -(1 + xd) ]  # Cambia el eje de referencia, y el lado derecho del canal se mide desde la izquierda
    x = round((xi + xd) / 2)  # Se calcula el promedio de ambas posiciones
    y1, y2 = ( start[x], stop[x],)  # Se determinan las coordenadas verticales de la constricción
    y1 = int(y1)
    y2 = int(y2)
    if box is not None:
        ymin, xmin, ymax, xmax = box
        # Genera los puntos a graficar
        p1 = (
            round(x * (xmax - xmin) / mask.shape[1]),
            round(y1 * (ymax - ymin) / mask.shape[0]),
        )
        p2 = (
            round(x * (xmax - xmin) / mask.shape[1]),
            round(y2 * (ymax - ymin) / mask.shape[0]),
        )
    else:
        # Genera los puntos a graficar
        p1 = (x, y1)
        p2 = (x, y2)
    return p1, p2

def chanel_a(wch, alpha):
    return wch / abs(np.tan(alpha))


def chanel_b(Rc, alpha):
    return Rc / abs(np.sin(alpha))


def chanel_r_lim(rho, alpha, wch):
    """Entradas: radio de curvatura del canal, su ángulo y su semiancho
    salidas: Radio límite de la célula antes de deformarse; determina que ecuación se utilizará para encontrar la posición inicial de la célula"""
    A = (rho * np.tan(alpha / 2)) ** 2
    B = (1 + np.cos(alpha)) ** 2
    C = (rho + wch) ** 2
    Rlim = -rho + (A * B + C) ** 0.5
    return Rlim
def cell_initial_pos(Rc, rho, alpha, wch):
    """Entradas: Radio inicial de la célula, radio de curvatura del canal, ángulo del canal y semiancho de la constricción
    Salidas: Posición inicial de la célula, calculada a partir de la geometría del canal y la célula

    Calcula la posicion inicial de la célula a partir de las ecuaciones planteadas en el artículo de referencia XXXXX
    La ecuacion utilizada depende de si el radio de curvatura del canal es menor o mayor al radio límite de la célula
    """
    Rlim = chanel_r_lim(rho, alpha, wch)
    a = chanel_a(wch, alpha)
    b = chanel_b(Rc, alpha)
    if Rc > Rlim:
        x0 = Rc - (b - a)
    else:
        A = (Rc + rho) ** 2
        B = (rho - wch) ** 2
        C = rho * np.tan(alpha / 2)
        x0 = Rc - (-C + (A - B) ** 0.5)

    return x0


def chanel_contour(mask, box, s=1):
    """Entradas: Máscara de un canal y la caja de la detección, opcionalmente se puede elegir el número de pixeles a ignorar
    Salidas: Coordenadas de los puntos del contorno del canal, reescalados a las dimensiones de la imagen original
    Encuentra el contorno de un canal a partir de la máscara de la predicción, y entrega las coordenadas de los puntos de este"""

    mask = (mask > 0.5).astype( np.uint8)  # ignora puntos con puntuación baja en la predicción
    ymin, xmin, ymax, xmax = box
    # Se calucula el ancho y alto de la caja y se reescalan las coordenadas
    h = (ymax - ymin) / mask.shape[0]
    w = (xmax - xmin) / mask.shape[1]
    top_contour = mask.shape[0] - np.argmax(mask, axis=0)
    y = top_contour * h
    x = np.arange(len(top_contour)) * w

    x = x[s - 1 : -s]
    y = y[s - 1 : -s]
    return x, y

def chanel_alpha( mask, box, s=7, o=1,):
    """Entradas: Máscara de un canal y la caja de la detección, opcionalmente se puede elegir el número de pixeles a ignorar y el desplazamiento del borde del canal
    Salidas: Ángulo del canal, calculado a partir de la máscara de la predicción, en radianes"""

    mask = (mask > 0.5).astype( np.uint8)  # ignora puntos con puntuación baja en la predicción
    x, y = chanel_contour(mask, box, s=s)
    canal_edge = np.argmin(y.astype(np.uint8))

    x1, x2 = x[canal_edge - o - 1], x[canal_edge]
    y1, y2 = y[canal_edge - o - 1], y[canal_edge]
    slope = (y2 - y1) / (x2 - x1)
    alpha = np.arctan(slope)

    return alpha

def bboxInference(image_files, bboxModelDir, threshold=0.5):
    """Entradas: Lista de archivos de imagen, directorio del modelo de detección de cajas delimitadoras y umbral de confianza
    
    Salidas: Lista de resultados de detección de cajas delimitadoras, cada uno con las coordenadas de la caja, la clase, el recorte y si está ocupado
    Realiza la inferencia de detección de cajas delimitadoras en las imágenes proporcionadas, extrayendo las coordenadas de las cajas, las clases y los recortes de las imágenes.
    La salida es una lista de diccionarios, cada uno representando los resultados de una imagen.
    Las claves de cada diccionario son:
    - 'boxes': Lista de coordenadas de las cajas detectadas.
    - 'classes': Lista de clases de las cajas detectadas.
    - 'cutouts': Lista de recortes de las imágenes correspondientes a las cajas detectadas.
    - 'occupied': Número de cajas ocupadas (clase 2).
    - 'filename': Nombre del archivo de imagen procesado.
    - 'imgNum': Número de imagen procesada (índice en la lista de imágenes).

    """

    results = []
    bboxModel = tf.saved_model.load(bboxModelDir) #Carga el modelo de detección de cajas delimitadoras

    print(f'Running bbox inference on {len(image_files)} images')

    for  imgNum,img in enumerate(image_files):
        infer = bboxModel.signatures['serving_default']

        # Carga la imagen y la convierte a un array de numpy, para luego procesarla con el modelo
        input_tensor = load_image_into_numpy_array(img)
        detections = infer(tf.constant(input_tensor))
        tenPercentile = len(image_files) // 10
        if imgNum % tenPercentile == 0:
            print(f'BBox inference progress: {imgNum}/{len(image_files)} images processed')
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)

        height, width = input_tensor.shape[1:3]
        # Reescalando las coordenadas de las cajas a las dimensiones originales de la imagen
        for i in range(len(boxes)):
            boxes[i][0] = int(boxes[i][0] * height)
            boxes[i][1] = int(boxes[i][1] * width)
            boxes[i][2] = int(boxes[i][2] * height)
            boxes[i][3] = int(boxes[i][3] * width)
        boxes = np.array(boxes, dtype=np.uint16)
        scores = np.array(scores, dtype=np.float32)
        classes = np.array(classes, dtype=np.uint8)

        # Genera un diccionario para almacenar los resultados de la imagen actual
        imgResults = {
        'boxes': [],
        'classes': [],
        'cutouts': [],
        'occupied': 0,
        'filename': img,
        'imgNum': imgNum
        }
    

        # Para cada imagen que supere el umbral de confianza, se extrae la caja, la clase y el recorte
        # Los resultados se almacenan en el diccionario imgResults
        for i, box in enumerate(boxes):
            if scores[i] > threshold:
                ymin, xmin, ymax, xmax = box
                cutout = input_tensor[0, ymin:ymax, xmin:xmax, :]
                imgResults['boxes'].append(box)
                imgResults['classes'].append(classes[i])
                if classes[i] == 2:
                    imgResults['cutouts'].append(upscale_image(cutout,(128,128)))
                    imgResults['occupied'] += 1

                else:
                    imgResults['cutouts'].append(None)
        sorted_indices = np.argsort([box[0] for box in imgResults['boxes']])


        # Ordena las cajas, clases y recortes por la coordenada y de la caja (ymin), de tal manera que
        # se pueda identificar el número del canal según el orden de las cajas en diferentes instantes
        imgResults['boxes'] = [imgResults['boxes'][i] for i in sorted_indices]
        imgResults['classes'] = [imgResults['classes'][i] for i in sorted_indices]
        imgResults['cutouts'] = [imgResults['cutouts'][i] for i in sorted_indices]
        results.append(imgResults)
    del bboxModel
    print('BBox inference completed.')
    return results

def separateExperiments(imgNumList):
    """Entradas: Lista de números de imagen
    Salidas: Lista de grupos de números de imagen, donde cada grupo contiene números consecutivos
    
    Agrupa los números de imagen en grupos de números consecutivos, para identificar segmentos continuos de imágenes que representan la deformación de una célula individual"""
    if not imgNumList:
        return []
    
    groups = []
    currentGroup = [imgNumList[0]]
    
    for num in imgNumList[1:]:
        # Comprueba si el número actual es consecutivo al último número del grupo actual, añadiéndolo al grupo si es así
        if num == currentGroup[-1] + 1:
            currentGroup.append(num)
        # Si no es consecutivo, se cierra el grupo actual y se inicia uno nuevo
        else:
            groups.append(currentGroup)
            currentGroup = [num]
    
    # Añade el último grupo si no está vacío, lo que puede ocurrir cuando el último frame contiene una célula.
    if currentGroup:
        groups.append(currentGroup)
    
    return groups


def visualizeBBoxes(img, inference):
    """Entradas: Ruta de la imagen y resultados de la inferencia de cajas delimitadoras
       Perimite visualizar la imagen original con las cajas delimitadoras dibujadas sobre ella."""
    original_image = cv2.imread(img)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    for idx, box in enumerate(inference['boxes']):
        ymin, xmin, ymax, xmax = box
        cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Draw rectangle
        cv2.putText(original_image, str(idx), (xmax + 5,(ymax + ymin)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 120, 10), 2)  # Add index

    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    plt.title("Original Image with Bounding Boxes")
    plt.axis("off")
    plt.show()

    



def visualizeMask(normalized_image, mask, box=None, threshold = None ,save = False, output_path='', isChannel = False, score=None):
    """Entradas: Imagen normalizada, máscara de la predicción, caja de la detección, umbral opcional, si se desea guardar la imagen y la ruta de salida
       Permite visualizar la imagen original con la máscara de la predicción superpuesta"""

    if threshold is not None:
        mask = mask > threshold
    # Blend the mask with the original image
    print(f"mask shape: {mask.shape}")
    print(f"normalized_image shape: {normalized_image.shape}")
    
    blended_image = normalized_image[0].numpy() * 0.7 + mask * [201/255, 12/255, 10/255] * 0.3

    P1, P2 = None, None
    if isChannel:
        P1, P2 = find_constrict_center(mask, box, s=15)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Original image
    axes[0].imshow(normalized_image[0])
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")

    # Blended image with mask
    axes[1].imshow(blended_image)
    if P1 is not None and P2 is not None:
        axes[1].plot([P1[0], P2[0]], [P1[1], P2[1]], color='red', linewidth=2)
    title = "Imagen con Máscara Superpuesta"
    if score is not None:
        title += f'_score_{score}'
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    if save:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def classification(classModelDir, results, batchSize = 32,threshold = 0.45):
    """ Entradas: Directorio del modelo de clasificación, resultados de la detección de cajas delimitadoras en forma de lista de diccionarios, tamaño del lote y umbral de confianza
        Salidas: Diccionario con las células clasificadas por canal, donde cada clave es el
        número del canal y el valor es una lista de diccionarios con los recortes, nombre del archivo, número de imagen, caja y si está ocupado.


        """
    print(f'Running classification on {len(results)} images')
    classModel = tf.keras.models.load_model(classModelDir)


    # Se inicializa un diccionario donde cada clave es el número del canal según su posicion de arriba a abajo
    # y el valor es una lista de diccionarios con los recortes, nombre del archivo, número de imagen, caja y el número de canales ocupados en ese frame
    numCh = len(results[0]['boxes'])
    cells = {i: [] for i in range(numCh)}
    

    classificationBatches = []
    for r in results:
        for cNum,c in enumerate(r['cutouts']):
            if c is not None:
                classificationBatches.append(
                {
                    'cutout': c,
                    'filename': r['filename'],
                    'imgNum': r['imgNum'],
                    'box': r['boxes'][cNum],
                    'channel': cNum,
                    'occupied': r['occupied'],
                }
            )


    # Separa los recortes en lotes para procesarlos en paralelo
    classificationBatches = [classificationBatches[i:i + batchSize] for i in range(0, len(classificationBatches), batchSize)]
    print(f'Number of batches: {len(classificationBatches)}')

    # Se procesa cada lote de recortes, obteniendo las puntuaciones de clasificación
    for batchNum,batch in enumerate(classificationBatches):
        cutouts = [item['cutout'] for item in batch]
        cutouts = np.array(cutouts)
        print(f'Processing batch {batchNum + 1}/{len(classificationBatches)}')
        scores = classModel.predict(cutouts)
        

        # Si el puntaje de clasificación supera el umbral, se añade el recorte al diccionario de células
        # Se utiliza el número del canal para clasificar las células en el diccionario
        for i, item in enumerate(scores):
            if item[0] > threshold:
                cells[batch[i]['channel']].append(batch[i])



    del classModel
    print('Classification completed.')
    return cells



def getExperimentList(channelData):
    """Entradas: Diccionario con las células clasificadas por canal, donde cada clave es el número del canal y el valor es una lista de diccionarios con los recortes, nombre del archivo, número de imagen, caja y si está ocupado.
       Salidas: Lista de experimentos, donde cada experimento es un diccionario con las mismas claves que el diccionario de células, pero con un número de experimento asignado.
       
       Agrupa las imágenes de células según segmentos continuos, asignando un número de experimento a cada grupo de imágenes consecutivas."""

    experimentsList = []
    expNum = 1
    # Revisa el diccionario de cada canal
    for chNum in range(len(channelData)):
        channel = channelData[chNum]
        # Si se detecta alguna célula en el canal, se procede a agrupar las imágenes
        if len(channel) > 0:
            # Se extraen los índices de las imágenes de cada célula detectada en el canal
            imgNumList = [item['imgNum'] for item in channel]
            cells = {}
            # Los datos de la detección se guardan temporalmente en un diccionario donde la clave es el número de imagen
            for item in channel:
                cells[item['imgNum']] = item
            # El diccionario se divide según los segmentos continuos de números de imagen
            separatedGroups = separateExperiments(imgNumList)
            
            # Se añade a una lista de experimentos un diccionario con el mismo formato que el de las células, pero con un número de experimento asignado
            # Se debe recalcar que la lista contiene los datos de un instante en particular para facilitar la agrupación de lotes.
            # Los experimentos se agruparán nuevamente una vez que se hayan segmentado las células
            for group in separatedGroups:
                for imgNum in group:
                    cells[imgNum]['expNum'] = expNum
                    experimentsList.append(cells[imgNum])
                expNum += 1
    return experimentsList


def cellSegmentation(segmentationModelDir, experimentsList, batchSize=32, displacementThreshold=5, threhold=0.55):
    """ Entradas: Directorio del modelo de segmentación, lista de experimentos, tamaño del lote, umbral de desplazamiento y umbral de segmentación
        Salidas: Lista de experimentos con las máscaras de segmentación de las células, donde cada experimento es un diccionario con las mismas claves que la lista de experimentos original, pero con una máscara añadida."""
    print(f'Running cell segmentation on {len(experimentsList)} experiments')
    segmentationModel = tf.keras.models.load_model(segmentationModelDir)

    # Se divide la lista de experimentos en lotes para procesarlos en paralelo
    batchedExperimentsList = [experimentsList[i:i+batchSize] for i in range(0, len(experimentsList), batchSize)]
    print(f'Number of batches: {len(batchedExperimentsList)}')

    maskResults = []
    #Se realiza la inferencia de segmentación en cada lote de experimentos
    #Los resultados obtenidos se añaden a una lista de diccionarios que tienen el mismo formato que la lista de experimentos original, pero con una máscara añadida
    for batchNum,batch in enumerate(batchedExperimentsList):
        print(f'Processing batch {batchNum + 1}/{len(batchedExperimentsList)}')
        cellArray = np.array([item['cutout'] for item in batch])
        normalized_image = tf.image.convert_image_dtype(cellArray, tf.float32) #Normaliza las imágenes a un rango de [0, 1]
        masks = segmentationModel.predict(normalized_image)


        for i, item in enumerate(batch):
            item['mask'] = keep_largest_component(masks[i])  #Elimina detecciones pequeñas y ruidosas de la máscara
            maskResults.append(item)


    # Se agrupan los resultados de segmentación por número de experimento, para facilitar el análisis posterior
    # Se crea un diccionario donde la clave es el número de experimento y el valor es una lista de diccionarios con los resultados de segmentación de ese experimento
    distinct_exp_nums = set(item['expNum'] for item in maskResults)
    experiments = {num: [] for num in distinct_exp_nums}
    for item in maskResults:
        experiments[item['expNum']].append(item)


    # Para cada experimento, se calcula el borde derecho de la célula y el radio de la célula
    expOutput = []
    for expNum, expList in experiments.items():
        expData = {
        'files': [item['filename'] for item in expList],
        'channel': expList[0]['channel'],
        'occupied': expList[0]['occupied'],
        'Rc': 0.0,  # Default value, will be updated later
        'filenames': []
    }
        rightCellEdge = []
        for item in expList:
            box = np.array(item['box'], dtype=np.uint16)
            mask = item['mask']

            w_scale = (box[3] - box[1]) / 128
            h_scale = (box[2] - box[0]) / 128

            cols = np.any(mask > threhold, axis=0)  # Check for non-zero values along columns
            rightEdge = mask.shape[1] - np.argmax(cols[::-1]) - 1  # Last non-zero column
            rightEdge = rightEdge * w_scale + box[1]
            rightCellEdge.append(rightEdge)

            expData['filenames'].append(item['filename'].split('.')[0].split('_')[-1] )

            if expData['Rc'] == 0.0:
                expData['firstFrame'] = item['cutout']
                expData['Rc'] = np.sqrt(np.sum(mask > threhold) * w_scale * h_scale / np.pi)
                expData['firstFrameBox'] = item['box']
            
    

        # Se asegura que la máscara no decresca, lo cual puede ser producido por el ruido en la segmentación, ya que la célula no puede retroceder en un escenario real
        for r in range(len(rightCellEdge)):
            if r > 0:
                rightCellEdge[r] = max(rightCellEdge[r], rightCellEdge[r-1])
        expData['rightCellEdge'] = np.array(rightCellEdge)

        # Los elementos que no sean células se asumen rígidos, por lo que se pueden filtrar 
        # si se mantiene en una posición fija en el tiempo, que en este caso se determina
        # cuando la mediana de la posición es muy cercana a los extremos del desplazamiento,
        # indicando que el elemento no se ha movido significativamente
        positionArray = np.array(rightCellEdge)
        lastPosition, firstPosition = positionArray[-1], positionArray[0]
        medianPosition = np.median(positionArray)
        displacementFromMedian  = min(np.abs(lastPosition - medianPosition), np.abs(firstPosition - medianPosition))
        if displacementFromMedian > displacementThreshold*w_scale:
            expOutput.append(expData)
    print('Cell segmentation completed.')
    return expOutput

def keep_largest_component(mask, threshold=0.5):
    """
    Filtra una máscara binaria para conservar solo el objeto más grande.
    
    Args:
        mask (numpy.ndarray): Máscara binaria de entrada (0 = fondo, 1 = objeto).
    
    Returns:
        numpy.ndarray: Máscara con solo el componente conectado más grande.
    """
    # Etiquetar componentes conectados
    labeled_mask = label(mask> threshold)
    
    # Calcular propiedades de las regiones
    regions = regionprops(labeled_mask)
    
    if len(regions) == 0:  # Si no hay objetos
        return np.zeros_like(mask)
    
    # Encontrar la región más grande por área
    largest_region = max(regions, key=lambda x: x.area)
    
    # Crear máscara solo con el objeto más grande
    largest_mask = np.zeros_like(mask)
    largest_mask[labeled_mask == largest_region.label] = 1
    
    return largest_mask

def segmentChannelAndSaveReults(channelModelDir, outputDir, expOutput, startIndex = 1, threshold=0.15):
    """Entradas: Directorio del modelo de segmentación de canales, directorio de salida, lista de experimentos y número de índice inicial
       Salidas: Lista de metadatos de los experimentos, donde cada diccionario contiene el número de experimento, el radio de la célula, el número de canales ocupados en
       el primer frame, el ancho del canal, y los archivos que contienen el primer y último frame procesado."""
    print(f'Segmenting channels and saving results for {len(expOutput)} experiments')
    channelModel = tf.keras.models.load_model(channelModelDir)
    metadataList = []

    # El proceso se realiza para cada experimento en la lista de experimentos
    for expNum, exp in enumerate(expOutput):
        box = exp['firstFrameBox']
        frame = exp['firstFrame']
        frame = np.array(frame)


   
        h_scale = (box[2]-box[0]) / 128

        if frame.ndim == 3:
            frame = np.expand_dims(frame, axis=0)  # Añade una dimensión de batch si no existe

        normalized_image = tf.image.convert_image_dtype(frame, tf.float32)  # Normaliza la imagen a un rango de [0, 1]
        mask = channelModel.predict(normalized_image)
        mask = np.squeeze(mask)  # Remueve la dimensión de batch si es necesario
        mask = keep_largest_component(mask, threshold=threshold)  # Elimina detecciones pequeñas y ruidosas de la máscara
        # Calcula el ancho de la constricción y el ángulo de entrada
        P1, P2 = find_constrict_center(mask, box = None, s=8)
        wc = (P2[1] - P1[1]) * h_scale
        alpha = np.abs(chanel_alpha(mask, box, s=15, o = 13))
        

        # Guarda la ruta del primer y último frame procesado
        firstFrameNumber = exp['files'][0]
        lastFrameNumber = exp['files'][-1]


        # Encuentra el valor del borde izquierdo de la constricción para tomarlo como punto de referencia
        x,y = chanel_contour(mask, box, s=15)
        edge = x[np.argmin(y)] + box[1]

        # No se logra encontrar un método adecuado para calcular el radio de curvatura de la constricción, por lo que se asume su valor
        rho = 7.0

        # Se calcula la posición de la célula en el momento en el que impacta con la constricción
        x0 = cell_initial_pos(exp['Rc'], rho, alpha, wc/2)

        # Se calcula el avance de la célula en cada instante
        AL =  exp['rightCellEdge'] - edge - x0

        #Para cada experimento se añanden las propiedades geométricas del canal encontradas 
        metadata = {
            'expNum': expNum + startIndex,
            'Rc': exp['Rc'],
            'occupied': exp['occupied'],
            'wc': wc,
            'firstFrame': firstFrameNumber.split('\\')[-1],
            'lastFrame': lastFrameNumber.split('\\')[-1],
        }
        metadataList.append(metadata)

        # Se guardan los resultados
        output_txt_path = os.path.join(outputDir, f'res_exp_{expNum+startIndex}.txt')
        with open(output_txt_path, 'w') as f:
            for idx, alValue in enumerate(AL):
                # f.write(f"{idx/50:3f} {alValue:.4f} {exp['Rc']} {wc/2} {exp['occupied']} {exp['filenames'][idx]} \n")
                f.write(f"{idx/50:3f} {alValue:.4f}\n")
            
        print(f"TXT saved to {output_txt_path}")

    return metadataList





# Valores por defecto (backup si no hay archivo config.txt)
DEFAULT_CONFIG = {
    "classModelDir": r".\models\classification\ResNet101_classification.keras",
    "bboxModelDir": r".\models\BBox\Mobilenet_BBox\saved_model",
    "channelModelDir": r".\models\segmentation\channelModel.keras",
    "segmentationModelDir": r".\models\segmentation\migratedUnet.keras",
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
    config = DEFAULT_CONFIG.copy()  # Carga los valores por defecto
    
    if not os.path.exists(config_path):
        print("No config file found, using default paths.")
        return config

    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignora líneas vacías y comentarios (que empiezan con #)
            if not line or line.startswith('#'):
                continue
            # Divide en key: value (solo si hay ":")
            if ':' in line:
                key, value = line.split(':', 1)  # Divide solo en el primer ":"
                key = key.strip()
                value = value.strip()
                
                # Convierte números a int
                if key in ["startImgNum", "endImgNum", "batchSize"]:
                    try:
                        value = int(value)
                    except ValueError:
                        print(f"Warning: Invalid number for {key}. Using default.")
                        value = DEFAULT_CONFIG[key]
                
                config[key] = value

    return config