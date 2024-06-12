import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def create_finetune_model(input_shape, regs, n_classes, drop_rate = 0.1):
    model = keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), input_shape=(*input_shape, 1), kernel_regularizer=regs, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))

    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=regs))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))

    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu',  kernel_regularizer=regs))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))

    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu',  kernel_regularizer=regs))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))

    model.add(Flatten())

    model.add(Dense(512, activation='relu',  kernel_regularizer=regs))
    model.add(Dropout(drop_rate))

    model.add(Dense(n_classes, activation='softmax',  kernel_regularizer=regs))

    return model

def get_model(weight_file):
    regularizer = keras.regularizers.L1(0.0001)
    model = create_finetune_model((256, 256), regularizer, 3)
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer='adam'
    )

    model.load_weights(weight_file)

    return model