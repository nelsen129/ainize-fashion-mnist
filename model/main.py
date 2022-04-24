import tensorflow as tf
import tensorflow.keras as keras

IMAGE_DIM = 28
NUM_CLASSES = 10


def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.
    test_images = test_images / 255.
    return (train_images, train_labels), (test_images, test_labels)


def load_model(channels=128):
    inputs = keras.Input([IMAGE_DIM, IMAGE_DIM])
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(channels, activation='relu')(x)
    x = keras.layers.Dense(NUM_CLASSES)(x)
    outputs = x
    return keras.Model(inputs, outputs)


def train(output_path, batch_size, num_train_epochs):
    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = load_model()
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=num_train_epochs,
              batch_size=batch_size)
    model.save(output_path)


def main():
    train('Fashion-MNIST/model.h5', 32, 10)


if __name__ == '__main__':
    print(tf.__version__)
    main()
