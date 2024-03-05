import tensorflow as tf
from keras.datasets import mnist
from keras import layers, models
from keras.utils import to_categorical
import pandas as pd


def build_model(num_classes):
    input_layer = layers.Input(shape=(28, 28, 1))
    conv_layer_1 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
    maxpooling_layer_1 = layers.MaxPooling2D(pool_size=(2, 2))(conv_layer_1)

    conv_layer_2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(maxpooling_layer_1)
    maxpooling_layer_2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_layer_2)

    conv_layer_3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(maxpooling_layer_2)
    maxpooling_layer_3 = layers.MaxPooling2D(pool_size=(2, 2))(conv_layer_3)

    flatten_layer = layers.Flatten()(maxpooling_layer_3)
    dense_layer_1 = layers.Dense(units=flatten_layer.shape[1], activation='relu', kernel_regularizer='l2')(flatten_layer)
    output_layer = layers.Dense(num_classes, activation='softmax')(dense_layer_1)

    model = models.Model(inputs=[input_layer], outputs=[output_layer])

    return model


if __name__ == "__main__":
    df = pd.read_csv("data/raw/A_Z Handwritten Data.csv")
    df.rename(columns={'0':'label'}, inplace=True)
    df = df.sample(frac=1, random_state=42)

    x_train = df.drop('label', axis=1).to_numpy().reshape(-1, 28, 28, 1) / 255.0
    # y_train = to_categorical(df['label'].to_numpy())
    y_train = df['label']
    num_classes = y_train.nunique()

    del df
    
    y_train = to_categorical(y_train.to_numpy())

    model = build_model(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.2, shuffle=True)

    model.save("models/en_alphabet_model")