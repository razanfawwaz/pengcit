import streamlit as st
import cv2
import numpy as np

# Fungsi untuk mendeteksi objek manusia dalam gambar menggunakan YOLO
def detect_human(image):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    height, width, channels = image.shape

    # Membuat blob dari gambar (praproses)
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Menyimpan informasi deteksi objek manusia
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Objek dengan class_id 0 adalah manusia
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return indexes, boxes

# Fungsi untuk mengaburkan objek manusia dalam gambar
def main(image, indexes, boxes):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            roi = image[y:y + h, x:x + w]
            roi = cv2.GaussianBlur(roi, (15, 15), 0)
            image[y:y + h, x:x + w] = roi
    return image

# Tampilan Streamlit
st.title("Deteksi dan Blur Objek Manusia")
uploaded_image = st.file_uploader("Unggah gambar")

if uploaded_image is not None:
    image = cv2.imread(uploaded_image)
    st.image(image, channels="BGR", caption="Gambar Asli")

    if st.button("Deteksi dan Blur Objek Manusia"):
        indexes, boxes = detect_human(image.copy())
        result_image = main(image.copy(), indexes, boxes)
        st.image(result_image, channels="BGR", caption="Gambar dengan Objek Manusia yang Di-blur")
