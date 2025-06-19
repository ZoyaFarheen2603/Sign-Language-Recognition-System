import os
import pickle
import mediapipe as mp
import cv2

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'  # FATAL

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Setup MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the data directory
DATA_DIR = './data'

# Initialize lists to hold data and labels
data = []
labels = []

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: The directory {DATA_DIR} does not exist.")
    exit()

# Iterate over each class directory in the data directory
for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue  # Skip non-directory files

    # Iterate over each image in the class directory
    for img_path in os.listdir(class_dir):
        img_file_path = os.path.join(class_dir, img_path)
        if not os.path.isfile(img_file_path):
            continue  # Skip if not a file

        data_aux = []
        x_ = []
        y_ = []

        # Read and process the image
        img = cv2.imread(img_file_path)
        if img is None:
            print(f"Error: Unable to read image {img_file_path}.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Process the hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Save the collected data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data collection and saving complete.")
