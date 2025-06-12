import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('tea_leaf_disease_model_20250612_131822.h5')

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ⚠️ Do NOT apply optimizations or quantization
# (This keeps the model's original float32 weights and structure)

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
