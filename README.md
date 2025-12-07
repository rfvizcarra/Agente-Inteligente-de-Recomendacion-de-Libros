# **📚 Agente Inteligente de Recomendación de Libros**

## **📋 Descripción del Proyecto**

Este proyecto implementa un **Agente Inteligente de Recomendación** basado en Deep Learning utilizando Python y TensorFlow/Keras.

A diferencia de un modelo de regresión tradicional, este agente no se limita a predecir una calificación aislada. Su objetivo principal es **simular un proceso de toma de decisiones**: el agente analiza un conjunto de libros candidatos para un usuario específico, predice el nivel de satisfacción para cada uno y **recomienda activamente** la mejor opción (Top-1 Recommendation).

El sistema utiliza una **arquitectura híbrida** que combina:

1. **Filtrado Colaborativo Neural (NCF):** Aprende patrones latentes de las preferencias históricas del usuario y su interacción con los libros.  
2. **Filtrado Basado en Contenido (NLP):** Utiliza procesamiento de lenguaje natural para analizar semánticamente las descripciones de los libros, permitiendo recomendar títulos basándose en su temática y no solo en su popularidad.  
1. 

## **🗂️ Dataset**

El conjunto de datos original consta de dos fuentes principales:

* books\_data.csv: Metadatos de los libros (título, descripción, autores, categorías).  
* Books\_rating.csv: Interacciones de usuarios (User\_ID, Book\_ID, Score).

Dataset obtenido del website Kaggle: *https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data*

**Estadísticas del Procesamiento:**

* Se cargó una muestra inicial de 100,000 registros para optimizar el rendimiento.  
* **Limpieza:** Se eliminaron columnas irrelevantes (imágenes, links) y filas con valores nulos en campos críticos.  
* **Datos Finales:** El dataset limpio para entrenamiento consta de aproximadamente **68,930 muestras** de alta calidad.

## **🛠️ Metodología y Pipeline**

### **1\. Preprocesamiento de Datos**

* **Estandarización:** Normalización de títulos (minúsculas y guiones bajos) para garantizar la consistencia en la combinación de datos (Merge).  
* **Codificación de Etiquetas (Label Encoding):** Transformación de User\_id y Title a secuencias de números enteros únicos para ser procesados por los Embeddings.  
* **Procesamiento de Lenguaje Natural (NLP):**  
  * Tokenización de descripciones (Vocabulario: 10,000 palabras más frecuentes).  
  * Padding de secuencias a una longitud fija de 100 palabras.

### **2\. Arquitectura del Modelo (Deep Learning)**

Se diseñó una red neuronal con la API Funcional de Keras que consta de tres ramas de entrada:

1. **Vector de Usuario:** Input (1) \- \- \- \> Embedding (Dim 16\) \- \- \- \> Flatten.  
2. **Vector de Libro:** Input (1) \- \- \- \> Embedding (Dim 16\) \- \- \- \> Flatten.  
3. **Vector de Descripción:** Input (100) \- \- \- \>Text Embedding (Dim 16\) \- \- \- \> Global Average Pooling.

Fusión: Los tres vectores  se concatenan y pasan por capas densas (Dense 64 \- \- \- \> Dense 16\) con activación ReLU para aprender relaciones no lineales complejas.

Salida: Una neurona con activación lineal (Regresión) para predecir el puntaje (1.0 a 5.0).

### **📉 Análisis de Mejora del Modelo: Combatiendo el Overfitting**

#### **1\. El Problema Detectado: Overfitting (Sobreajuste)**

En la primera versión del modelo, observamos un comportamiento clásico de **Sobreajuste** con una brecha muy alta entre entrenamiento y validación.

* **Evidencia Numérica:**  
  * **Training Loss (Entrenamiento):** Cayó hasta **\~0.15**.  
  * **Validation Loss (Prueba):** Se estancó en **\~1.25**.  
  * **La Brecha:** Existía una diferencia de **1.1 puntos**. Esto indica que el modelo estaba "memorizando" el set de entrenamiento casi a la perfección, pero fallaba notablemente al generalizar.  
* **Diagnóstico:** El modelo tenía demasiada capacidad. En lugar de aprender patrones generales, estaba memorizando cada calificación individual.

#### **2\. La Solución: Estrategias de Regularización**

Para solucionar esto, aplicamos tres técnicas de regularización diseñadas para "dificultar" el aprendizaje y forzar al modelo a generalizar:

* **Regularización L2 (Ridge):** Añadimos `kernel_regularizer=l2(0.01)` para penalizar pesos grandes, obligando a la red a buscar patrones más simples.  
* **Capas de Dropout (Abandono):** Insertamos `Dropout(0.5)` para apagar aleatoriamente el 50% de las neuronas en cada paso, evitando la co-dependencia.  
* **Reducción de Complejidad:** Redujimos los Embeddings (de 32 a 16\) y las capas “Dense” (de 128 a 64\) para limitar la "capacidad de memoria" del modelo.

#### **3\. El Resultado: Estabilización y Robustez**

Tras aplicar estos cambios, las curvas de aprendizaje mostraron una mejora crítica en la estabilidad:

* **Cierre de la Brecha (Gap):**  
  * **Training Loss Actual:** Subió a **\~1.30** (ya no memoriza).  
  * **Validation Loss Actual:** Se situó en **\~1.58**.  
  * **Mejora:** La brecha se redujo de 1.1 a solo **0.28**. Las líneas ahora se mueven juntas.  
* **Trade-off (Costo-Beneficio):**  
  * El **RMSE** subió ligeramente de **1.08** a **1.12**.  
  * *Interpretación:* Aunque el error numérico es un poco más alto, el modelo es **honesto**. Un RMSE de 1.12 con una brecha pequeña es infinitamente mejor que un RMSE de 1.08 logrado mediante "trampa" (memorización), ya que el nuevo modelo funcionará de manera predecible con usuarios reales.

## **📊 Resultados y Evaluación**

El modelo fue entrenado durante un maximo de 10 Epochs con un tamaño de batch de 32\. Se agregó un Early Stop para mejorar el resultado y ayudar a bajar el Overfitting.

* **Métrica de Evaluación:** RMSE (Root Mean Squared Error).  
* **Interpretación:** El RMSE indica, en promedio, qué tan alejada está la predicción del modelo (en estrellas) respecto a la calificación real del usuario.

*Ejemplo de inferencia del Agente:*

Plaintext  
\--- Agente de Recomendación \---  
Libro: dr.\_seuss:\_american\_icon  
Rating Real del Usuario: 5.0 estrellas  
Predicción del Agente: 4.82 estrellas  
Veredicto del Agente: ¡Altamente Recomendado\! 🌟

## **💻 Tecnologías Utilizadas**

* **Lenguaje:** Python 3.x  
* **Librerías Principales:**  
  * TensorFlow / Keras: Construcción y entrenamiento de la red neuronal.  
  * Pandas: Manipulación y limpieza de datos.  
  * Scikit-Learn: Preprocesamiento (LabelEncoder) y división de datos (Train/Test Split).  
  * Matplotlib: Visualización de curvas de aprendizaje.

## **🚀 Instrucciones de Ejecución**

1. Clonar el repositorio.  
2. Asegurarse de tener los archivos CSV en la ruta especificada (sample\_data/ y drive/MyDrive/project\_data/).  
3. Instalar las dependencias necesarias:  
   pip install pandas numpy tensorflow scikit-learn matplotlib  
4. Ejecutar el notebook o script principal.

---

*Proyecto desarrollado como parte del curso de Agentes Inteligentes.*

