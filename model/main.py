import keras
import tensorflow as tf

from data import load_data

IMAGE_DIM = 28
NUM_CLASSES = 10


def load_model(channels=128):
    inputs = keras.Input([IMAGE_DIM, IMAGE_DIM])

    x = keras.layers.Reshape([IMAGE_DIM, IMAGE_DIM, 1])(inputs)
    x = keras.layers.Conv2D(channels // 4, 3, activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv2D(channels // 2, 3, activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv2D(channels, 3, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(channels, activation='relu')(x)
    x = keras.layers.Dense(NUM_CLASSES)(x)

    outputs = x
    return keras.Model(inputs, outputs)


def train(output_path, batch_size, num_train_epochs):
    train_dataset, test_dataset = load_data(batch_size)
    model = load_model()
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_dataset, validation_data=test_dataset, epochs=num_train_epochs,
              batch_size=batch_size)
    model.save(output_path)


def main():
    train('Fashion-MNIST/model.h5', 128, 50)


if __name__ == '__main__':
    print(tf.__version__)
    main()
