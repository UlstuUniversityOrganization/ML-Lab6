import tensorflow as tf
from keras.datasets import mnist
from keras import layers, models
import pandas as pd
import gradio as gr
from PIL import Image
import numpy as np


model = tf.keras.models.load_model('models/mnist_model')


def recognize_digit(image):
    if image is not None:
        image = image['layers'][0]
        image_pil = Image.fromarray(image)
        image_resized = image_pil.resize((28, 28), Image.BILINEAR)

        image_np = np.array(image_resized).reshape((1, 28, 28, 1)).astype('float32') / 255.0
        prediction = model.predict(image_np, verbose=0)
        return {str(i): float(prediction[0][i]) for i in range(10)}
    return ''


if __name__ == '__main__': 
    interface = gr.Interface(fn=recognize_digit, 
                             inputs=gr.Paint(container=True, image_mode='L'), 
                             outputs=gr.Label(num_top_classes=10), 
                             live=True)
    
    interface.launch()