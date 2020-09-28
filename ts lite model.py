#  display meta (label data) within a tensor flow lite model
import os
import tensorflow as tf

#
# export_model_path = '/home/pi/birdclass/ssd_mobilenet_v1_1_metadata_1.tflite'
# export_model_path = '/home/pi/birdclass/'
# displayer = metadata.MetadataDisplayer.with_model_file(export_model_path)
# export_json_file = os.path.join(export_directory,
#                     os.path.splitext(model_basename)[0] + ".json")
# json_file = displayer.get_metadata_json()
# # Optional: write out the metadata as a json file
# with open(export_json_file, "w") as f:
#   f.write(json_file)

import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/pi/birdclass/ssd_mobilenet_v1_1_metadata_1.tflite")
interpreter.allocate_tensors()

input = interpreter.tensor(interpreter.get_input_details()[0]["index"])
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
for i in range(10):
  input().fill(3.)
  interpreter.invoke()
  print("inference %s" % output())

