import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle


@st.experimental_singleton
def load_model():
    return tf.keras.models.load_model('final_model.h5')


@st.experimental_singleton
def index_to_class():
    with open('index_to_class.pickle', 'rb') as f:
        out = pickle.load(f)
    return out


def get_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    st.image(img)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(img)

model = load_model()
idx_to_cls = index_to_class()

st.write("""
# Flutter Finder
### Indentify 100 butterfly and moth species.
""")

image_file = st.file_uploader('Upload image of a butterfly or moth', type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    image = get_image(image_file)
    predictions = model.predict(image)
    predict_cls_inx = np.argmax(predictions, axis=1)[0]
    predict_cls = idx_to_cls[predict_cls_inx]
    st.write(f'**Predicted class:** {predict_cls}')

st.write("""
## Project Goal
Build a tool to identify Butterflies and Moths

## Methodology
Develop a multiclass image classification model using labelled images of Butterflies and Moths

## Data
Data for model training was downloaded from Kaggle ([Butterfly & Moths Image Classification 100 species](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)).
It has 100 categories of butterflies and moths. All the images are of size 224 X 224 X 3. Training dataset has 12639 
images with 101 to 187 images in each category. Validation and test datasets has 5 images from each category.  
""")

st.image(Image.open('different_categories.png'))

st.write("""
## Model training
1. Tensorflow was used as the model training library
2. Loaded pre trained resnet50 model to perform transfer learning. 
3. Dense layers along with Dropout layers were added to train the initial model. Validation dataset accuracy: 94.0%
5. Model training was done with image augmentation to reduce overfitting. Didn't observe any improvement in validation accuracy
6. Fine tuned the pretrained model weights from the start of '7 by 7 convolution layers'. Validation dataset accuracy: 94.4%
7. Test dataset accuracy: 94.8% 

## Explore the code
[flutter-finder](https://github.com/nis89mad/flutter-finder)
""")
