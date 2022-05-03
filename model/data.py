import tensorflow as tf
import tensorflow_addons as tfa
import keras.datasets.fashion_mnist as fashion_mnist

IMAGE_DIM = 28


def rotate_image(image, rg=0.5):
    angle = tf.random.uniform([1], -rg, rg)
    return tfa.image.rotate(image, angle)


def translate_image(image, rg=0.1):
    trans = tf.random.uniform([2], int(IMAGE_DIM * -rg), int(IMAGE_DIM * rg))
    return tfa.image.translate(image, trans)


def invert_image(image):
    if tf.random.uniform([1], 0, 1) > 0.5:
        return 1 - image
    else:
        return image


def color_jitter_image(image, rg=0.2):
    return image + tf.random.uniform([1], -rg, rg)


def augment_data(dataset):
    dataset = dataset.map(lambda x, y: (rotate_image(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (translate_image(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (invert_image(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (color_jitter_image(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def load_data(batch_size):
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = tf.cast(train_images / 255., tf.float32)
    test_images = tf.cast(test_images / 255., tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(60000)
    train_dataset = augment_data(train_dataset)
    train_dataset = train_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
    return train_dataset, test_dataset
