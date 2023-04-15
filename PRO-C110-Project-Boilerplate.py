import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')

# Attach the camera indexed as 0
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

    # Read a frame from the camera
    status, frame = camera.read()

    # If we were successfully able to read the frame
    if status:

        # Flip the frame
        frame = cv2.flip(frame, 1)

        # Resize the frame
        resized_frame = cv2.resize(frame, (224, 224))

        # Expand the dimensions of the frame
        expanded_frame = np.expand_dims(resized_frame, axis=0)

        # Normalize the frame before feeding it to the model
        normalized_frame = expanded_frame / 255.0

        # Get predictions from the model
        predictions = model.predict(normalized_frame)

        # TODO: Process the predictions as needed

        # Display the frames captured
        cv2.imshow('feed', frame)

        # Wait for 1ms
        code = cv2.waitKey(1)

        # If the space key is pressed, break the loop
        if code == 32:
            break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()
