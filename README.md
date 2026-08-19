## Adding-Sunglasses-to-Your-Passport-Photo-Using-OpenCV.
## AIM
Adding-Sunglasses-to-Your-Passport-Photo-Using-OpenCV.

## Software Used
Anaconda – Python 3.7

Jupyter Notebook / VS Code

OpenCV (cv2)

NumPy

Matplotlib

## input
## Step 1 — Import libraries
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
## Step 2 — Load your passport photo
```
img = cv2.imread("passport.jpg")


# OpenCV uses BGR; convert to RGB for displaying with Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


plt.imshow(img_rgb)
plt.axis("off");
```
## Step 3 — Detect the face
```
OpenCV includes a Haar Cascade classifier:

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(100, 100)
)


print("Faces detected:", len(faces))
```
## Step 4 — Load sunglasses
```

Ideally, use a PNG sunglasses image with a transparent background:

glasses = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)


print(glasses.shape)

The fourth channel is the alpha/transparency channel.
Step 5 — Position sunglasses

For each detected face:

for (x, y, w, h) in faces:


    # Approximate position of the eyes
    glasses_x = x + int(0.10 * w)
    glasses_y = y + int(0.35 * h)


    glasses_width = int(0.80 * w)


    scale = glasses_width / glasses.shape[1]
    glasses_height = int(glasses.shape[0] * scale)


    glasses_resized = cv2.resize(
        glasses,
        (glasses_width, glasses_height)
    )
```
## Step 6 — Overlay the sunglasses
```
    if glasses_resized.shape[2] == 4:


        alpha = glasses_resized[:, :, 3] / 255.0


        sunglasses_rgb = glasses_resized[:, :, :3]


        y1 = glasses_y
        y2 = min(glasses_y + glasses_height, img.shape[0])


        x1 = glasses_x
        x2 = min(glasses_x + glasses_width, img.shape[1])


        overlay = sunglasses_rgb[:y2-y1, :x2-x1]
        alpha = alpha[:y2-y1, :x2-x1]


        for c in range(3):
            img[y1:y2, x1:x2, c] = (
                alpha * overlay[:, :, c]
                + (1 - alpha) * img[y1:y2, x1:x2, c]
            )
```
## Step 7 — Display the result
```
result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(6, 6))
plt.imshow(result)
plt.axis("off");
```
## Step 8 — Save the output
```
cv2.imwrite("passport_with_sunglasses.jpg", img)
print("Saved successfully!")
```
## OUTPUT
<img width="484" height="637" alt="image" src="https://github.com/user-attachments/assets/d05ddf72-21f2-41e5-a16a-f62dbf7ed457" />

<img width="483" height="638" alt="image" src="https://github.com/user-attachments/assets/7de90635-7228-49d2-8f46-6ac89f7da221" />

