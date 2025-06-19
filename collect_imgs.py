import os
import cv2

# Set the directory where the data will be stored
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 25  # Number of classes
dataset_size = 100  # Number of images per class

# Open the video capture (camera) with device index 0
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for the user to press 'q' to start collecting data
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        cv2.putText(frame, 'Ready? Press "Q" to start', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'Starting to capture images for class {j}. Press "q" to stop early.')

    # Capture dataset_size number of images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

        counter += 1
        # Use a small delay to allow the camera to adjust
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print(f'Stopping early for class {j}. Collected {counter} images.')
            break

cap.release()
cv2.destroyAllWindows()
