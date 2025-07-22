import cv2
import os
import time
from datetime import datetime

# === CONFIGURATION ===
CAPTURED_IMG_DIR = 'captured_test_data'
IMAGES_PER_LABEL = 100
CAPTURE_INTERVAL = 0.125    # seconds between captures
PREP_DELAY = 10             # seconds before starting to capture each label
CAPTURE_ALL_LABELS = False  # Set to False to capture only a specific label
SPECIFIC_LABEL = 'A'        # Change this to the label you want to capture if not capturing all

img_size = 200              # size for saved images

# load labels from example directory
asl_example_dir = 'example_data'
labels = sorted([
    d for d in os.listdir(asl_example_dir) 
    if os.path.isdir(os.path.join(asl_example_dir, d))
])

# Filter labels if only one is needed
if not CAPTURE_ALL_LABELS:
    if SPECIFIC_LABEL not in labels:
        raise ValueError(f"Label '{SPECIFIC_LABEL}' not found in {asl_example_dir}")
    labels = [SPECIFIC_LABEL]

# === SETUP ===
os.makedirs(CAPTURED_IMG_DIR, exist_ok=True)
for label in labels:
    os.makedirs(os.path.join(CAPTURED_IMG_DIR, label), exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Starting auto-capture. Press 'q' to quit.")

label_index = 0

# Pre-select a single example image for each label
example_images = {}
for label in labels:
    example_dir = os.path.join(asl_example_dir, label)
    example_imgs = [f for f in os.listdir(example_dir) if f.lower().endswith(('.jpg', '.png'))]
    if example_imgs:
        # Pick an image for each label
        example_img_path = os.path.join(example_dir, example_imgs[0])
        example_img = cv2.imread(example_img_path)
        if example_img is not None:
            example_img = cv2.resize(example_img, (img_size, img_size))
            example_images[label] = example_img

# Start capturing images for each label
while label_index < len(labels):
    current_label = labels[label_index]

    # === PREP PHASE ===
    prep_start = time.time()
    while time.time() - prep_start < PREP_DELAY:

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't read frame. Exiting ...")
            break

        frame = cv2.flip(frame, 1)
        display_frame = cv2.resize(frame.copy(), (img_size * 4, img_size * 4))

        # Show the pre-selected example image for the current label
        example_img = example_images.get(current_label)
        if example_img is not None:
            display_frame[0:img_size, -img_size:] = example_img

        # display countdown and label info
        countdown = PREP_DELAY - int(time.time() - prep_start)
        cv2.putText(display_frame, f"Get ready for: {current_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Starting in {countdown}s", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, "Example Image", (display_frame.shape[1] - img_size, img_size + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('ASL Data Collector', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # === CAPTURE PHASE ===
    print(f"Capturing images for label: {current_label}")
    last_capture_time = 0
    captured = 0

    # Capture images until we reach the desired count for this label
    while captured < IMAGES_PER_LABEL:

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = cv2.resize(frame.copy(), (img_size * 4, img_size * 4))
        resized_frame = cv2.resize(frame, (img_size, img_size))

        # Show current label and captured count
        cv2.putText(display_frame, f"Label: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Images: {captured}/{IMAGES_PER_LABEL}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('ASL Data Collector', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # Check if it's time to save the image
        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{current_label}_{timestamp}.jpg"
            filepath = os.path.join(CAPTURED_IMG_DIR, current_label, filename)
            cv2.imwrite(filepath, resized_frame)
            print(f"Saved: {filepath}")
            captured += 1
            last_capture_time = current_time

    print(f"Finished capturing for: {current_label}\n")
    label_index += 1

print("Data collection complete.")
cap.release()
cv2.destroyAllWindows()