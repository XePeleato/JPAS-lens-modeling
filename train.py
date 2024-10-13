import numpy as np
import glob

from keras import Sequential
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from astropy.io import fits
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence


class FitsDataGenerator(Sequence):
    def __init__(self, fits_files, labels, batch_size, dim, n_channels, shuffle=True):
        self.fits_files = fits_files
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.fits_files))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.fits_files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        fits_files_temp = [self.fits_files[k] for k in indexes]

        X, y = self.__data_generation(fits_files_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.fits_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, fits_files_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, file in enumerate(fits_files_temp):
            X[i,] = read_and_normalize_fits(file)
            y[i] = self.labels[i]

        return X, y


def read_fits_image(fits_path):
    with fits.open(fits_path) as hdul:
        image_data = hdul[1].data
    return image_data


def read_and_normalize_fits(fits_path):
    with fits.open(fits_path) as hdul:
        image_data = hdul[1].data
        # Normalizar la imagen para que sus valores estén en el rango [0, 1]
        normalized_image = image_data / np.max(image_data)
    return normalized_image



fits_directory = 'conjunto_entrenamiento/'

fits_files = sorted(glob.glob(fits_directory + '*.fits'))

# Suponiendo que todas las imágenes tienen el mismo tamaño!!
num_bands = len(fits_files)
image_shape = read_fits_image(fits_files[0]).shape

multi_band_image = np.zeros((image_shape[0], image_shape[1], num_bands))

for i, fits_file in enumerate(fits_files):
    multi_band_image[:, :, i] = read_and_normalize_fits(fits_file)

multi_band_image = np.transpose(multi_band_image, (2, 0, 1))

print("Dimensiones del tensor multibanda:", multi_band_image.shape)

# 1 = lente, 0 = no lente
labels = [1, 1, 1, 1]

# Dividir en entrenamiento y prueba
X_train, X_temp, y_train, y_temp = train_test_split(multi_band_image, labels, test_size=0.3, random_state=42, stratify=labels)

# Dividir en validación y prueba
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Forma de los conjuntos de entrenamiento, validación y prueba:")
print(X_train.shape, X_val.shape, X_test.shape)





# Generador de datos
batch_size = 32
dim = (512, 512)
n_channels = 1 # Número de bandas

training_generator = FitsDataGenerator(X_train, y_train, batch_size, dim, n_channels)
validation_generator = FitsDataGenerator(X_val, y_val, batch_size, dim, n_channels)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(9200, 9200, n_channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Para clasificación binaria

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=20
)
