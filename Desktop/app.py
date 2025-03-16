import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("your_model.h5")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame (resize, normalize, etc.)
    input_frame = cv2.resize(frame, (224, 224)) / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)
    
    # Make prediction
    prediction = model.predict(input_frame)
    
    # Display output
    cv2.putText(frame, f"Prediction: {np.argmax(prediction)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
