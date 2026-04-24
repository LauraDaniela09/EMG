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

**visualizacion de la señal**

```python
 ==============================
# RECORTE AUTOMÁTICO (ZONA ÚTIL)
# ==============================
if len(segments) > 0:
    inicio = max(0, segments[0][0] - int(1*fs))
    fin = min(len(emg_filt), segments[-1][1] + int(1*fs))
else:
    inicio = 0
    fin = len(emg_filt)

# ==============================
# GRÁFICA PRINCIPAL MEJORADA
# ==============================
plt.figure(figsize=(14,5))

# Señal EMG (más fina)
plt.plot(t[inicio:fin], emg_filt[inicio:fin],
         color='black', linewidth=1, label='Señal EMG')

# Envolvente (más visible)
plt.plot(t[inicio:fin], envelope_smooth[inicio:fin],
         color='magenta', linewidth=2, label='Envolvente suavizada')

# ==============================
# SOMBREADO DE CONTRACCIONES
# ==============================
for (start, end) in segments:
    if start >= inicio and end <= fin:
        plt.axvspan(t[start], t[end],
                    color='purple', alpha=0.25)

# ==============================
# ESTILO FINAL
# ==============================
plt.title("Contracciones musculares detectadas (EMG)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(alpha=0.3)

plt.show()
```
<img width="1167" height="470" alt="image" src="https://github.com/user-attachments/assets/c9a49069-6354-4674-bf4e-e83884891bd4" />

Esta parte del código se encarga de mostrar de forma individual las contracciones musculares detectadas en la señal electromiográfica (EMG).
Primero, se define un margen de 200 milisegundos antes y después de cada contracción (`margen = int(0.2 * fs)`) para visualizar con mayor detalle el inicio y el final de cada una. Luego, se seleccionan las cinco primeras contracciones (`regiones = regiones[:5]`) con el fin de limitar el número de gráficas generadas.

Mediante un ciclo `for`, el programa recorre cada contracción detectada y genera una figura independiente. En cada una, se grafica la señal EMG en color fucsia, representando la variación de amplitud del músculo durante la contracción, mientras que el intervalo correspondiente a la actividad muscular se resalta con un sombreado lila para identificar claramente el momento de la contracción.

```python
for i, (s, e) in enumerate(segments):

    m = int(0.2 * fs)

    s2 = max(0, s - m)
    e2 = min(len(emg_filt), e + m)

    plt.figure(figsize=(10,3))
    plt.plot(t[s2:e2], envelope_smooth[s2:e2], color='black')

    plt.axvspan(t[s], t[e], color='purple', alpha=0.25)

    plt.title(f"Contracción {i+1}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(alpha=0.3)
    plt.show()
```

<img width="855" height="316" alt="image" src="https://github.com/user-attachments/assets/9b829bd4-245d-4221-bf3e-62b9944ad301" />
<img width="870" height="316" alt="image" src="https://github.com/user-attachments/assets/549f3e7f-de91-441f-8a44-2dc9de80b2b4" />
<img width="855" height="316" alt="image" src="https://github.com/user-attachments/assets/7655c05f-1d78-4cca-a067-8e6dbd8fbfcb" />
<img width="855" height="316" alt="image" src="https://github.com/user-attachments/assets/bc0987ee-3b7d-45bb-bae6-5bd99dda898f" />
<img width="855" height="316" alt="image" src="https://github.com/user-attachments/assets/85ea867b-eb98-4c5d-9b1e-cacf098f17a4" />

**calculo de frecuencia media y mediana**

```python
from scipy.signal import welch

def calc_freq(seg, fs):
    f, Pxx = welch(seg, fs=fs, nperseg=min(256, len(seg)))

    if np.sum(Pxx) == 0:
        return np.nan, np.nan

    f_media = np.sum(f * Pxx) / np.sum(Pxx)

    cumsum = np.cumsum(Pxx)
    idx = np.where(cumsum >= cumsum[-1]/2)[0]

    f_mediana = f[idx[0]] if len(idx) > 0 else np.nan

    return f_media, f_mediana

# calcular para cada contracción
resultados = []

for i, (ini, fin) in enumerate(segments[:5]):
    seg = signal_f[ini:fin]
    fm, fmed = calc_freq(seg, fs)
    resultados.append((i+1, fm, fmed))

import pandas as pd
df = pd.DataFrame(resultados, columns=["Contracción", "F_media", "F_mediana"])
print(df)
```
Este código calcula la frecuencia media y mediana de cada contracción detectada en la señal EMG. Primero, se utilizan las librerías `scipy.signal`, `numpy` y `pandas` para procesar los datos. Dentro de un ciclo `for`, se recorre cada contracción y se aplica el método de Welch para obtener el espectro de potencia de la señal, representado por las frecuencias (`f`) y su energía (`Pxx`). Luego, se normaliza el espectro para calcular la frecuencia media, que indica el promedio ponderado de las frecuencias, y la frecuencia mediana, que marca el punto donde se concentra el 50 % de la energía total. Finalmente, los resultados se organizan en una tabla que muestra, para cada contracción muscular, los valores obtenidos de ambas frecuencias, facilitando el análisis de la variación en el contenido espectral de la señal.

<img width="354" height="131" alt="Captura de pantalla 2026-04-23 201203" src="https://github.com/user-attachments/assets/7c7ab969-9191-41ba-b92d-42374c65a4d2" />


**analisis de las frecuencias**

Al analizar la evolución de la frecuencia media y la frecuencia mediana a lo largo de las contracciones simuladas, se observa que ambas se mantienen prácticamente constantes, sin variaciones significativas entre una contracción y otra. La frecuencia media presenta valores entre 11.58 Hz y 11.61 Hz, mientras que la frecuencia mediana se mantiene alrededor de 7.58 Hz a 7.60 Hz. Esta estabilidad indica que no hay evidencia de fatiga muscular simulada, ya que en una señal EMG real la frecuencia mediana tiende a disminuir progresivamente con el tiempo cuando existe fatiga. Por lo tanto, los resultados obtenidos sugieren que las contracciones analizadas corresponden a una actividad muscular constante y controlada, sin cambios notables en las características espectrales de la señal.

```python
plt.figure()

plt.plot(df["Contracción"], df["F_media"], marker='o', color='#e91e63', label="Media")
plt.plot(df["Contracción"], df["F_mediana"], marker='s', color='#6e6e6e', label="Mediana")

plt.title("Evolución de frecuencias")
plt.xlabel("Contracción")
plt.ylabel("Frecuencia (Hz)")
plt.legend()
plt.grid()

plt.show()
```

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/0d60eaa6-1e09-4e1a-a999-edfbbe18c6d8" />


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


<h1 align="center"><i><b>𝐏𝐚𝐫𝐭𝐞 C 𝐝𝐞𝐥 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨</b></i></h1>

```python
fft_results = []

for (start, end) in segments_limpios:
    seg = filtered[start:end]
    if len(seg) == 0:
        continue

    N = len(seg)
    fft_vals = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(N, 1/fs)

    fft_results.append((freqs, fft_vals))

print("FFT calculadas:", len(fft_results))

# =========================
# 7. ESPECTROS (ROSADO + PICO)
# =========================
if len(fft_results) > 0:
    n = len(fft_results)
    cols = 3
    rows = math.ceil(n / cols)

    plt.figure(figsize=(15,10))

    for i, (freqs, fft_vals) in enumerate(fft_results):
        idx = np.argmax(fft_vals)

        plt.subplot(rows, cols, i+1)
        plt.plot(freqs, fft_vals, color='deeppink')
        plt.plot(freqs[idx], fft_vals[idx], 'ko')
        plt.xlim(0,500)
        plt.title(f"Contracción {i+1}")
        plt.grid()

    plt.tight_layout()
    plt.show()

```

<img width="985" height="297" alt="image" src="https://github.com/user-attachments/assets/8352bdd5-941f-4cee-b310-16697a413f21" />


```python

if len(fft_results) > 1:
    plt.figure(figsize=(10,5))

    f1, fft1 = fft_results[0]
    f2, fft2 = fft_results[-1]

    plt.plot(f1, fft1, color='dimgray', linewidth=2, label='Inicio')
    plt.plot(f2, fft2, color='deeppink', linewidth=3, label='Fatiga')

    plt.xlim(0,500)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.title("Comparación espectral")
    plt.legend()
    plt.grid()
    plt.show()
```

<img width="881" height="471" alt="image" src="https://github.com/user-attachments/assets/883e01e7-bf70-416d-8d93-b73326836313" />


```python
if len(fft_results) > 0:
    picos = [f[np.argmax(fft)] for f, fft in fft_results]

    plt.figure(figsize=(8,5))
    plt.plot(picos, color='deeppink', marker='o', linewidth=2)

    plt.title("Desplazamiento del pico espectral")
    plt.xlabel("Contracción")
    plt.ylabel("Hz")
    plt.grid()
    plt.show()
```
<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/01bb3ca1-297e-4679-a8c6-505b37a7dd5d" />

<h1 align="center"><i><b>Conclusiones</b></i></h1>

**Comparación espectral:**

Al comparar los espectros de las primeras y últimas contracciones, se observa una disminución del contenido de altas frecuencias en las contracciones finales. Además, el espectro presenta un desplazamiento hacia frecuencias más bajas, lo cual es característico del fenómeno de fatiga muscular.

**Reducción de alta frecuencia:**

La reducción de componentes de alta frecuencia se asocia con la disminución en la velocidad de conducción de las fibras musculares durante la fatiga.
