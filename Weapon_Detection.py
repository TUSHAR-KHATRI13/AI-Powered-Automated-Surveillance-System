import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import tkinter as tk
from tkinter import filedialog


net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]

output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Function to send email alert
def send_email_alert(image):
    # Email configurations
    sender_email = "tusharkhatri1008@gmail.com"
    sender_password = "Tushar Khatri" 
    receiver_email = "tsec.t12.lab@gmail.com"

    # Create message container
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Weapon Detected!"

    # Email body
    body = "A weapon has been detected in the frame. Please take appropriate action."
    msg.attach(MIMEText(body, 'plain'))

    image_attachment = MIMEImage(image)
    image_attachment.add_header('Content-Disposition', 'attachment', filename='weapon_detected.jpg')
    msg.attach(image_attachment)

    # Connect to SMTP server and send email
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp_server:
        smtp_server.starttls()
        smtp_server.login(sender_email, sender_password)
        smtp_server.send_message(msg)

# Loading image
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        start_processing(file_path)

def start_webcam():
    cap = cv2.VideoCapture(0)
    process_video(cap)

def start_processing(file_path):
    cap = cv2.VideoCapture(file_path)
    process_video(cap)

def process_video(cap):
    while True:
        _, img = cap.read()
        if not _:
            print("Error: Failed to read a frame from the video source.")
            break
        height, width, channels = img.shape
        
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layer_names)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Send email alert if weapon is detected
        if len(indexes) > 0:
            print("Weapon detected in frame. Sending email alert...")
            # Convert the image to bytes
            _, buffer = cv2.imencode('.jpg', img)
            image_bytes = buffer.tobytes()
            send_email_alert(image_bytes)
            
        # Draw rectangles and labels on detected objects
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        # Display image
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Create the GUI
root = tk.Tk()
root.title("Weapon Detection Model")

# Function to close the GUI properly
def close_gui():
    root.quit()
    root.destroy()

# Add heading
heading = tk.Label(root, text="Weapon Detection Model", font=("Helvetica", 16, "bold"), pady=10)
heading.pack()

# Add labels and buttons
label = tk.Label(root, text="Choose an option:", font=("Helvetica", 12))
label.pack()

select_file_button = tk.Button(root, text="Select File", command=select_file, font=("Helvetica", 12))
select_file_button.pack(pady=5)

start_webcam_button = tk.Button(root, text="Start Webcam", command=start_webcam, font=("Helvetica", 12))
start_webcam_button.pack(pady=5)

# Function to adjust button size when fullscreen
def resize(event):
    if root.state() == 'zoomed':
        select_file_button.config(font=("Helvetica", 16))
        start_webcam_button.config(font=("Helvetica", 16))
    else:
        select_file_button.config(font=("Helvetica", 12))
        start_webcam_button.config(font=("Helvetica", 12))

# Bind resize event
root.bind("<Configure>", resize)

# Add a quit button
quit_button = tk.Button(root, text="Quit", command=close_gui, font=("Helvetica", 12))
quit_button.pack(pady=10)

root.mainloop()
