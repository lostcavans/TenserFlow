import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt

# Cargar el modelo entrenado (si ya lo tienes guardado)
model = load_model('image_classifier.h5')  # Descomenta esta línea si has guardado tu modelo previamente

# Directorio donde están las imágenes de test
test_dir = 'dataset/test/'  # Cambia esto a la ruta de tu conjunto de datos de test

# Preprocesamiento de las imágenes de test
test_datagen = ImageDataGenerator(rescale=1./255)  # Escala las imágenes entre 0 y 1

# Crear generador de datos de test
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),  # Ajusta el tamaño según tu modelo
    batch_size=32,
    class_mode='categorical'  # Cambia esto si usas otro tipo de clasificación (binary, etc.)
)

# Evaluar el modelo en los datos de test
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")  # Imprime la pérdida y la precisión del modelo en los datos de test

# Hacer predicciones con el modelo en las imágenes de test (opcional)
y_true = test_generator.classes  # Las etiquetas reales de test
y_pred = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)  # Predicciones del modelo
y_pred_classes = np.argmax(y_pred, axis=1)  # Convierte las predicciones a clases utilizando argmax

# Mostrar el reporte de clasificación
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))  # Imprime un reporte de clasificación

# Hacer predicción en una nueva imagen
img_path = 'dataset/test/class3/3.jpg'  # Cambia esto a la ruta de la imagen que quieres clasificar

# Cargar la imagen y redimensionarla
img = load_img(img_path, target_size=(150, 150))  # Carga la imagen y la redimensiona al tamaño esperado por el modelo

# Convertir la imagen en un arreglo de numpy
img_array = img_to_array(img)  # Convierte la imagen cargada a un arreglo numpy

# Expande las dimensiones para hacer que el modelo lo vea como un "batch" de imágenes
img_array = np.expand_dims(img_array, axis=0)  # Agrega una dimensión extra para crear un batch de tamaño 1

# Normalizar la imagen (como hiciste con los datos de entrenamiento)
img_array = img_array / 255.0  # Escala la imagen entre 0 y 1

# Hacer la predicción
predictions = model.predict(img_array)  # Realiza la predicción sobre la imagen

# Mostrar los resultados de la predicción
print(f"Predicciones: {predictions}")  # Imprime las probabilidades de cada clase para la imagen

# Si tienes nombres de las clases, puedes asociar la predicción con la clase:
class_names = ['class1', 'class2', 'class3']  # Cambia esto con los nombres reales de tus clases
predicted_class = class_names[np.argmax(predictions)]  # Obtiene la clase con la mayor probabilidad
print(f"Predicción final: {predicted_class}")  # Imprime la clase predicha

# Mostrar la imagen y la predicción con los resultados
plt.figure(figsize=(6, 6))  # Configura el tamaño de la figura
plt.imshow(img)  # Muestra la imagen
plt.title(f"Predicción: {predicted_class}\nProbabilidades: {predictions[0]}")  # Muestra el título con la predicción y probabilidades
plt.axis('off')  # Elimina los ejes de la imagen
plt.show()  # Muestra la imagen con el título y la predicción
