import cv2
import torch
from torchvision.models import mobilenet_v2
from torchvision import transforms
import os

# Live demo for ASL recognition using a pre-trained model

# === CONFIGURATION ===
MODEL_NAME = 'asl_mobilenetv2_fine_tuning_captured_data'
SAVE_VIDEO = False  # Set to True to save the video
VIDEO_NAME = 'asl_live_capture.avi'

# Load labels from example directory
asl_example_dir = 'example_data' 
labels = sorted([
    d for d in os.listdir(asl_example_dir) 
    if os.path.isdir(os.path.join(asl_example_dir, d))
])

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join('models', MODEL_NAME, f'{MODEL_NAME}.pth')
model = mobilenet_v2(num_classes=29)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


video_save_dir = 'saved_videos'

os.makedirs(video_save_dir, exist_ok=True)
video_path = os.path.join(video_save_dir, VIDEO_NAME)

# Setup video writer if saving is enabled
video_writer = None
if SAVE_VIDEO:
    # Get frame width and height from the capture
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20  # fallback to 20 if FPS is 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

print("Starting live ASL recognition. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't read frame. Exiting ...")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess frame
    input_tensor = preprocess(rgb_frame).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        output = model(input_tensor)
        # get predicted class
        pred = output.argmax(dim=1).item()

    # Display prediction on frame
    cv2.putText(frame, f'Prediction: {labels[pred]}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Live ASL Recognition', frame)

    # Save frame to video if enabled
    if SAVE_VIDEO and video_writer is not None:
        video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
print(f"Video saved to: {video_path}" if SAVE_VIDEO else "No video saved.")