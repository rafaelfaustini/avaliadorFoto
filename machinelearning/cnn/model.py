from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(filters=32, activation='relu', input_shape=(128, 128, 3), kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, activation='relu', kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=6, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255, shear_range=90,
    zoom_range=90, rotation_range=20)

gerador_teste = ImageDataGenerator(rescale = 1./255, shear_range=90,
    zoom_range=90, rotation_range=20)
                
base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (128, 128),
                                                           batch_size = 64,
                                                           class_mode = 'categorical')
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                                               target_size = (128, 128),
                                                               batch_size = 64,
                                                               class_mode = 'categorical')

model.fit(base_treinamento, steps_per_epoch= len(base_treinamento),
          epochs = 100, validation_data = base_teste,
          validation_steps = len(base_teste),
          verbose=1)
            
