# Laboratorio-4-EMG

<h2 align="center">𝙞𝙣𝙩𝙧𝙤𝙙𝙪𝙘𝙘𝙞ó𝙣</h2>
La señal electromiográfica (EMG) permite analizar la actividad eléctrica producida por los músculos durante la contracción. Mediante su procesamiento digital es posible identificar variaciones en la amplitud y frecuencia que reflejan el estado de fatiga muscular. En esta práctica se emplean herramientas computacionales para adquirir, filtrar y analizar señales EMG, observando cómo cambia su contenido espectral a lo largo de varias contracciones.

<h2 align="center">𝙤𝙗𝙟𝙚𝙩𝙞𝙫𝙤</h2>

Analizar señales electromiográficas emuladas y reales mediante técnicas de segmentación y análisis espectral, con el fin de calcular la frecuencia media y mediana y evaluar su relación con la aparición de la fatiga muscular.

<h2 align="center">𝙞𝙢𝙥𝙤𝙧𝙩𝙖𝙘𝙞ó𝙣 𝙙𝙚 𝙡𝙞𝙗𝙧𝙚𝙧𝙞𝙖𝙨</h2>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
```
Esa parte del código muestra la importación de librerías necesarias para el procesamiento y análisis de señales EMG:

`numpy` como `np` para realizar operaciones numéricas y manejo de arreglos;
`pandas` como `pd` para cargar y manipular los datos de la señal;
 `matplotlib.pyplot ` como  `plt` para graficar los resultados y visualizar las contracciones;
y las funciones  `butter `,  `filtfilt ` y  `welch ` del módulo  `scipy.signal ` para aplicar filtros digitales y obtener el análisis espectral de la señal.

<h1 align="center"><i><b>𝐏𝐚𝐫𝐭𝐞 A 𝐝𝐞𝐥 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨</b></i></h1>
