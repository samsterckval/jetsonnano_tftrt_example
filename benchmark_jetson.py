import numpy as np
import tensorflow as tf
from tensorflow import lite
import keras
from keras.models import load_model
from keras.applications.mobilenetv2 import preprocess_input, decode_predictions
import os
import PIL
import time
from tftrt_helper import FrozenGraph, TfEngine, TftrtEngine
execution_path = os.getcwd()

model_input_path = os.path.join(execution_path, 'MobileNetV2_ImageNet.h5')
#model_output_path = os.path.join(execution_path, 'MobileNetV2_ImageNet.tflite')
input_image_path = os.path.join(execution_path, 'images/magpie.jpg')

# Lets optimize the model for the Jetson's GPU
input_model = load_model(model_input_path)
frozenmodel = FrozenGraph(input_model, (224, 224, 3))
print('FrozenGraph build.')
model = TftrtEngine(frozenmodel, 1, 'FP16', output_shape=(1000))
#model = TftrtEngine(frozenmodel, 1, 'INT8', output_shape=(1000))
print('TF-TRT model ready to rumble!')

# Load and preprocess the image
input_image = PIL.Image.open(input_image_path)
input_image = np.asarray(input_image)
preprocessed = preprocess_input(input_image)
preprocessed = np.expand_dims(preprocessed, axis=0)
print('input tensor shape : ' + str(preprocessed.shape))



# This actually calls the inference
print("Warmup prediction")
output = model.infer(preprocessed)
print(decode_predictions(output))

time.sleep(1)

print("starting now (Jetson Nano)...")
s = time.time()
for i in range(0,250,1):
    output = model.infer(preprocessed)
e = time.time()
print('elapsed : ' + str(e-s))
