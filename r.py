import yolov5 as yl
from PIL import Image
import numpy as np
import streamlit as st

model = yl.load('./best.pt')

image = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
if image is not None:
    image = Image.open(image)
    image = np.array(image)

    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000

    results = model(image)

    results.save(save_dir='./Output/')

    st.image('./Output/image0.jpg') 