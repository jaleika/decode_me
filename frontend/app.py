import streamlit as st
import pandas as pd
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

import streamlit as st
import pandas as pd
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

import time

url = 'http://127.0.0.1:8000'

#Use the full page instead of a narrow central column
st.set_page_config(page_title='Decode me',
                page_icon = '☝',
                layout = 'wide',
                initial_sidebar_state = 'expanded')

tab1, tab2 = st.tabs(["Decode me", "Team"])

with tab1:

    imageLocation = st.empty()
    imageLocation.image('face.jpg', width = 900)
#    st.image('face.jpg', width = 900)


    st.markdown("""
    <style>
      [data-testid=stSidebar] {
        background-color: #5B5B5B;
    }
    </style>
    """,unsafe_allow_html=True)


    with st.sidebar:
        st.header("Decode the face")
        st.markdown('Upload a picture of a person or people')
        uploaded_image = st.file_uploader("Choose a face image", type = ['png', 'jpg'], accept_multiple_files=False)
        if uploaded_image:
            img_bytes = uploaded_image.getvalue()
            imageLocation.image(img_bytes, width = 900)
            if st.button(f'Get result'):

                #url_endpoint = f"{url}predict"
                img_bytes = uploaded_image.getvalue()
                #st.image(Image.open(uploaded_image))
                res = requests.post(url = url + "/predict", files = {'image': img_bytes})

                st.write(f"The mood of the picture is: {res.json()['mood']}")
                # TODO: change img_bytes in the following so that it contains colored boxes around the faces encoding the mood the emotion as text

                imageLocation.image(img_bytes, width = 900)


    st.sidebar.image('face4.jpg', use_column_width=True)

with tab2:
    col1, col2, col3 = st.columns((4, 4, 2))

    with col1:
        st.markdown('## Meet the Decode me Team')
        st.markdown('#### Natasha 🍉')
        #st.image('Mariia.png', width = 250)
        st.markdown('#')
        st.markdown('#### Isabella 🍭')
        #st.image('Lisa1.png', width = 250)

    with col2:
        st.markdown('#')
        st.markdown('#')
        st.markdown('#### Hayri 🎈')
        #st.image('Malory.png', width = 250)
        st.markdown('#')
        st.markdown('#### Isabel 🍹')
        #st.image('David1.png', width = 250)
