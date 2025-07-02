# Descripción

Este repositorio contiene los modelos y el programa desarrollados para la caracterización mecánica de células automática mediante el uso de modelos de deep learning en un dispositivo de citometría de deformabilidad basado en constricciones (cDC) [1]. El programa analiza las imágenes para obtener una curva de avance normalizado de una célula siendo deformada en una constricción de un dispositivo cDC, la cual posteriormente puede ser ajustada por una curva que viene determinada por tres parámetros viscoelásticos de un modelo heurístico [1,2].

# Instrucciones de uso

Se deben descargar los modelos entrenados desde el siguiente link https://drive.google.com/drive/folders/11neVBOzhFlyhHNpek1qoThky8fxbMKZN?usp=sharing


En el archivo config.txt se encuentran parámetros de entrada para el programa; estos incluyen las carpetas de modelos de regresión de bounding boxes, clasificación y segmentación, las carpetas de entrada y salida de datos, tamaño de lote, y umbrales para el puntaje de las inferencias realizadas. Se pueden escoger los modelos ingresando la ruta adecuada, y cambiar los valores de los puntajes de corte de tal manera que se adecuen a las necesidades del usuario.

Para ejecutar el programa se debe trabajar en un entorno con Python 3.9.18 y tensorflow 2.10.1. El programa fue desarrollado en Windows 11 con CUDA 11.2 y CUDNN 9.7 por lo que se recomienda trabajar con estas versiones para asegurar la compatibilidad. En el archivo requirements.txt se incluyen todas las librerías instaladas en el entorno en el momento de desarrollo y pruebas del programa.

El programa se ejecuta desde la consola con el siguiente comando:

```
python cdc_ImageAnalizer.py
```

Los resultados se guardan en la carpeta indicada por el archivo de configuración, y contienen información geométrica de las células y los canales del dispositivo cDC. 

Los resultados obtenidos se deben procesar con el programa fittingExperiment.py desde la consola; éste programa requiere que los resultados de cdc_ImageAnalizer.py estén guardados en la carpeta /Results, y que la carpeta /sim contenga resultados de simulaciones de interacción fluido estructura.

Tras realizar los ajustes necesarios, el programa entrega en la carpeta /Output los archivos csv con las curvas ajustadas del avance de la célula, los gráficos de los ajustes realizados, y el archivo viscoProperties.txt que contiene las propiedades viscoelásticas que mejor mejor ajustan la curva de avance normalizado en el tiempo.

Si se busca realizar un entrenamiento posterior, los dataset con las imágenes anotadas para cada modelo se incluyen en el siguiente link https://drive.google.com/drive/folders/1vAYIAt-48-t41KwZR3fXe99R1uIhMu5S?usp=sharing

# Referencias

[1] Abarca A. «Desarrollo y aplicación de técnicas de caracterización mecánicacelular basadas en deformación por contacto en dispositivos de microfluídica». PhD Thesis. Caminos, 2023.

[2] Matus N. «Desarrollo de una técnica de análisis automático de imágenes para la caracterización mecánica celular en dispositivos tipo cDC mediante Deep Learning». 2025.
