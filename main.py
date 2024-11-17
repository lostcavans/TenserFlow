import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configuración de directorios de datos (ruta de las imágenes de entrenamiento y validación)
train_dir = "dataset/train/"  # Directorio de imágenes de entrenamiento
validation_dir = "dataset/validation/"  # Directorio de imágenes de validación

# Preprocesamiento de datos para el conjunto de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalización de los píxeles para que estén entre 0 y 1
    rotation_range=20,  # Rotación aleatoria de las imágenes hasta 20 grados
    width_shift_range=0.2,  # Desplazamiento horizontal aleatorio de las imágenes
    height_shift_range=0.2,  # Desplazamiento vertical aleatorio de las imágenes
    shear_range=0.2,  # Aplicar cizallamiento (transformación geométrica)
    zoom_range=0.2,  # Zoom aleatorio en las imágenes
    horizontal_flip=True  # Volteo horizontal aleatorio de las imágenes
)

# Preprocesamiento de datos para el conjunto de validación (solo normalización)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Generador para cargar las imágenes de entrenamiento y aplicar las transformaciones
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Ruta del directorio de entrenamiento
    target_size=(150, 150),  # Redimensionar las imágenes a 150x150 píxeles
    batch_size=32,  # Número de imágenes procesadas en cada paso
    class_mode="categorical"  # Utilizado para clasificación multiclase
)

# Generador para cargar las imágenes de validación (sin aumentos, solo normalización)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,  # Ruta del directorio de validación
    target_size=(150, 150),  # Redimensionar las imágenes a 150x150 píxeles
    batch_size=32,  # Número de imágenes procesadas en cada paso
    class_mode="categorical"  # Utilizado para clasificación multiclase
)

# Construcción del modelo de red neuronal
model = models.Sequential([  # Crear un modelo secuencial
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # Capa convolucional 1 con 32 filtros
    layers.MaxPooling2D((2, 2)),  # Capa de max pooling para reducir las dimensiones
    layers.Conv2D(64, (3, 3), activation='relu'),  # Capa convolucional 2 con 64 filtros
    layers.MaxPooling2D((2, 2)),  # Capa de max pooling
    layers.Conv2D(128, (3, 3), activation='relu'),  # Capa convolucional 3 con 128 filtros
    layers.MaxPooling2D((2, 2)),  # Capa de max pooling
    layers.Flatten(),  # Aplanar las salidas de las capas convolucionales para pasar a las densas
    layers.Dense(128, activation='relu'),  # Capa densa con 128 neuronas y activación ReLU
    layers.Dense(train_generator.num_classes, activation='softmax')  # Capa de salida, número de clases según el generador
])

# Compilación del modelo con el optimizador, función de pérdida y métricas
model.compile(
    optimizer='adam',  # Optimizador Adam
    loss='categorical_crossentropy',  # Función de pérdida para clasificación multiclase
    metrics=['accuracy']  # Métrica de precisión
)

# Entrenamiento del modelo
history = model.fit(
    train_generator,  # Datos de entrenamiento
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Número de pasos por época
    epochs=10,  # Número de épocas de entrenamiento
    validation_data=validation_generator,  # Datos de validación
    validation_steps=validation_generator.samples // validation_generator.batch_size  # Pasos de validación por época
)

# Evaluación y visualización de resultados
acc = history.history['accuracy']  # Precisión en el entrenamiento
val_acc = history.history['val_accuracy']  # Precisión en la validación
loss = history.history['loss']  # Pérdida en el entrenamiento
val_loss = history.history['val_loss']  # Pérdida en la validación

epochs_range = range(len(acc))  # Rango de las épocas para graficar

# Graficar precisión y pérdida de entrenamiento y validación
plt.figure(figsize=(8, 8))

# Subgráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Accuracy')  # Precisión en el entrenamiento
plt.plot(epochs_range, val_acc, label='Validation Accuracy')  # Precisión en la validación
plt.legend(loc='lower right')  # Ubicación de la leyenda
plt.title('Training and Validation Accuracy')  # Título del gráfico

# Subgráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Loss')  # Pérdida en el entrenamiento
plt.plot(epochs_range, val_loss, label='Validation Loss')  # Pérdida en la validación
plt.legend(loc='upper right')  # Ubicación de la leyenda
plt.title('Training and Validation Loss')  # Título del gráfico

plt.show()  # Mostrar los gráficos

# Guardar el modelo entrenado en un archivo .h5
model.save('image_classifier.h5')

# Imprimir las clases detectadas y sus índices
print("Clases detectadas:", train_generator.class_indices)
