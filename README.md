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


<h1 align="center"><i><b>𝐏𝐚𝐫𝐭𝐞 B 𝐝𝐞𝐥 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨</b></i></h1>

Se define y diseña un filtro pasabanda tipo butterworth para usar más adelante.
```python
plt.figure(figsize=(18,5))
plt.plot(t, signal, color='deeppink')
plt.title("Señal EMG del paciente")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()
```
<img width="1482" height="470" alt="image" src="https://github.com/user-attachments/assets/5b683bfe-6246-470e-af56-27aade8eae81" />

Se aplica el filtro definido anteriormente como pasabanda entre 20-450 Hz.
Se grafica esta señal recortada y filtrada.
```python
from scipy.signal import butter, filtfilt

def filtro_pasabanda(signal, fs, lowcut=20, highcut=450, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    señal_filtrada = filtfilt(b, a, signal)
    
    return señal_filtrada

# Aplicar filtro
filtered = filtro_pasabanda(signal, fs)
```
<img width="1790" height="590" alt="image" src="https://github.com/user-attachments/assets/2031ec83-e5d2-47a7-892a-6fdbe5edfa31" />

Se usa find peaks para identificar los picos (contracciones) y se grafica nuevamente la señal, pero resaltando estos picos identificados para ver su distribución y que sean correctos.

```python
plt.figure(figsize=(18,5))
plt.plot(t, filtered, color='deeppink')

for i, (start, end) in enumerate(segments_limpios):
    
    seg = filtered[start:end]
    
    # índice del máximo dentro del segmento
    idx_max = np.argmax(seg)
    
    # índice real en la señal completa
    idx_real = start + idx_max
    
    # punto en el pico
    plt.plot(t[idx_real], filtered[idx_real], 'ko')  # punto negro
    
    # numeración
    plt.text(t[idx_real], filtered[idx_real], f'{i+1}', fontsize=10)

plt.title("Picos de cada contracción")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()
```

<img width="1482" height="470" alt="image" src="https://github.com/user-attachments/assets/1239df4b-324f-4faa-a13f-bbdbbdd34eb6" />


Se divide la señal entre cada una de sus contracciones para analizar individualmente.

```python
import math

n = len(segments_limpios)
cols = 3
rows = math.ceil(n / cols)

plt.figure(figsize=(15,10))

for i, (start, end) in enumerate(segments_limpios):
    
    seg = filtered[start:end]
    N = len(seg)
    
    fft_vals = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    # índice del pico en frecuencia
    idx_max = np.argmax(fft_vals)
    
    plt.subplot(rows, cols, i+1)
    plt.plot(freqs, fft_vals, color='deeppink')
    
    
    plt.plot(freqs[idx_max], fft_vals[idx_max], 'ko')
    
    plt.title(f"Contracción {i+1}")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.xlim(0, 500)
    plt.grid()

plt.tight_layout()
plt.show()
```

<img width="1489" height="990" alt="image" src="https://github.com/user-attachments/assets/d538f1c9-1039-4a2c-b991-42d2bc68e199" />

Luego se calcula de cada contración la frecuencia media y la frecuencia mediana 

```python
mean_freqs = []
median_freqs = []

for i, (start, end) in enumerate(segments_limpios):
    
    seg = filtered[start:end]
    N = len(seg)
    
    # FFT
    fft_vals = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    # Frecuencia media
    mean_f = np.sum(freqs * fft_vals) / np.sum(fft_vals)
    
    # Frecuencia mediana
    cumulative = np.cumsum(fft_vals)
    median_f = freqs[np.where(cumulative >= cumulative[-1]/2)[0][0]]
    
    mean_freqs.append(mean_f)
    median_freqs.append(median_f)

# Mostrar resultados
for i in range(len(mean_freqs)):
    print(f"Contracción {i+1}: Media = {mean_freqs[i]:.2f} Hz | Mediana = {median_freqs[i]:.2f} Hz")
```
<img width="545" height="232" alt="image" src="https://github.com/user-attachments/assets/2a32bff6-4856-4b5f-985c-a934aa50811e" />
<img width="850" height="471" alt="image" src="https://github.com/user-attachments/assets/9ebffb61-b54e-47ec-a18a-c7a5f9196f6d" />
