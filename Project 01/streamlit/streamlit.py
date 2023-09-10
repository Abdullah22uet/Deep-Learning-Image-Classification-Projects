import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO


st.set_option('deprecation.showfileUploaderEncoding',False)
st.title('Potato Disease Classifier')
st.text('Provide URL of image')

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('models/1')
    return model

with st.spinner('Loading Model into Memory'):
    model = load_model()
    
classes = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def scale(image):
    image = tf.cast(image,tf.float32)
    image /=255.0
    
    return tf.image.resize(image,[256,256])

def decode_img(image):
    img = tf.image.decode_jpeg(image,channels=3)
    img = scale(img)
    return np.expand_dims(img , axis=0)


path = st.text_input('Enter URL to classify...','https://www.potatogrower.com/Images/0513/EB%20lesions1_opt.jpeg')
if path is not None:
    content = requests.get(path).content
    
    # st.write("predicted class:")
    with st.spinner('classifying...'):
        label = np.argmax(model.predict(decode_img(content)),axis=1)
        confidence = round(100* np.max(model.predict(decode_img(content))))
        st.write(f"Predicted Class : {classes[label[0]]}")
        st.write(f"Confidence : {confidence}%")
        
    image = Image.open(BytesIO(content))
    st.image(image , caption='Classifying whether disease is (Early_Blight) or (Late_Blight) or whether potato plants are healthy' , use_column_width=True)