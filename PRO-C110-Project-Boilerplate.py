import cv2
import numpy as np
import tensorflow as tf  # Import TensorFlow modules

# Load your TensorFlow model
# Replace 'path_to_your_model' with the actual path to your trained model
model = tf.keras.models.load_model('keras_model.h5')

camera = cv2.VideoCapture(0)

while True:
    check, frame = camera.read()

    if check:
        frame = cv2.flip(frame, 1)

        # Resize the frame to the desired dimensions
        # Example resizing to 224x224
        target_size = (224, 224)
        frame_resized = cv2.resize(frame, target_size)

        # Expand dimensions to create a batch dimension
        frame_expanded = np.expand_dims(frame_resized, axis=0)

        # Normalize the frame before feeding it to the model
        # Assuming pixel values are in the [0, 255] range
        frame_normalized = frame_expanded / 255.0

        # Get predictions from the model
        predictions = model.predict(frame_normalized)

        # Assuming you want to display the predictions on the frame
        print('Prediction: ', predictions)

        # Display the captured frame
        cv2.imshow('Result', frame)

        code = cv2.waitKey(1)
        if code == 32:
            break

camera.release()
cv2.destroyAllWindows()
