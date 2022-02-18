# import better_exceptions
from keras.applications import ResNet50,MobileNetV2     # , InceptionResNetV2
from keras.layers import Dense
from keras.models import Model
from keras import backend as K


def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae


def get_model(model_name="MobileNetV2"):
    base_model = None

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
    #elif model_name == "InceptionResNetV2":
       # base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="avg")
    elif model_name == "MobileNetV2":
        base_model = MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=None, pooling="avg")
    
    prediction_emotion = Dense(units=3, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_emo")(base_model.output)

    model = Model(inputs=base_model.input, outputs=[prediction_emotion])
    #model = Model(inputs=base_model.input, outputs=prediction)

    return model


def main():
    model = get_model("InceptionResNetV2")
    model.summary()


if __name__ == '__main__':
    main()
