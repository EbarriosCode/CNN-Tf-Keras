import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

altura, longitud = 100, 100
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'

cnn = load_model(modelo)
cnn.load_weights(pesos)

def predict(file):
    x = load_img(file, target_size = (longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis = 0)
    
    arreglo = cnn.predict(x) ##[[1,0]]

    resultado = arreglo[0]
    respuesta = np.argmax(resultado)

    if respuesta == 0:
        print('Es un Perro')
    
    elif respuesta == 1:
        print('Es un Gato')

    return respuesta


predict('nombre de la imagen')