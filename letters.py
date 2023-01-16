import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils
import numpy as np
from io import BytesIO
from PIL import Image

@st.experimental_singleton
def load_model_from_path(suppress_st_warning=True):
    return load_model('ArabicLetters.h5')
test  = load_model_from_path()

st.title("Arabic Letters Recognizer")

# Specify canvas parameters in application
drawing_mode = 'freedraw'
stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 3)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color='white',
    background_color='black',
    width=256,
    height=256,
    drawing_mode=drawing_mode,
    display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    key="full_app",
)



im = Image.fromarray(canvas_result.image_data)
im.save("img.png")
image = image_utils.load_img('img.png', color_mode="grayscale", target_size=(32,32))

st.markdown('This is what the model sees: ')
st.image(image)

image = image_utils.img_to_array(image)
image = image.reshape(1,32,32,1)
prediction = test.predict(image)

labels ={   0: 'أ',
            1:'ب',
            2:'ت',
            3:'ث',
            4:'ج',
            5:'ح',
            6:'خ',
            7:'د',
            8:'ذ',
            9:'ر',
            10:'ز',
            11:'س',
            12:'ش',
            13:'ص',
            14:'ض',
            15:'ط',
            16:'ظ',
            17:'ع',
            18:'غ',
            19:'ف',
            20:'ق',
            21:'ك',
            22:'ل',
            23:'م',
            24:'ن',
            25:'هـ',
            26:'و',
            27:'ي'}

st.markdown(f"<h1 style='color: green; font-size:40px'>Prediction: {labels[np.argmax(prediction)]}</h1>", unsafe_allow_html=True)
st.markdown(f"<h1 style='color: green; font-size:30px'>Confidence: {prediction[0][np.argmax(prediction)]}</h1>", unsafe_allow_html=True)