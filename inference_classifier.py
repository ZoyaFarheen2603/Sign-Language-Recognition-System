import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model from the pickle file
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# Initialize the camera (Change the index as necessary)
cap = cv2.VideoCapture(0)  # Change to 1 or other indices if using a different camera

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()
# Initialize MediaPipe Handsaz
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for real-time detection
    min_detection_confidence=0.7,  # Increased confidence for better detection
    min_tracking_confidence=0.7  # Increased tracking confidence for better tracking
)
# Define the labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'HELLO', 6: 'G', 7: 'SORRY', 8: 'I LOVE YOU', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'THANK YOU', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'}

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Get the dimensions of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Initialize variables to hold landmark data
    x_ = []
    y_ = []

    # Check if hands are detected in the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract landmark coordinates and normalize them
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            if len(x_) > 0 and len(y_) > 0:
                x_min = min(x_)
                y_min = min(y_)
                x_max = max(x_)
                y_max = max(y_)
                data_aux = [(x - x_min) for x in x_] + [(y - y_min) for y in y_]

                # Ensure there is data for prediction
                if len(data_aux) == 42:  # Expecting 21 landmarks * 2 coordinates
                    # Predict the sign language letter
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_class = int(prediction[0])
                    predicted_character = labels_dict.get(predicted_class, '?')  # Handle unexpected classes

                    # Draw a bounding box around the hand
                    x1 = int(x_min * W) - 10
                    y1 = int(y_min * H) - 10
                    x2 = int(x_max * W) + 10
                    y2 = int(y_max * H) + 10

                    # Ensure bounding box is within frame limits
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W, x2)
                    y2 = min(H, y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

                    # Display prediction results on the frame
                    cv2.putText(frame, f'Predicted: {predicted_character}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    print("Data_aux length is not 42, cannot make a prediction.")
            else:
                print("No hand landmarks detected.")

    # Display the
    cv2.imshow('Hand Sign Language Detection', frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print('Inference complete.')
