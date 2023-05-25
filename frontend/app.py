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


#Use the full page instead of a narrow central column
st.set_page_config(page_title='Decode me',
                page_icon = '☝',
                layout = 'wide',
                initial_sidebar_state = 'expanded')

tab1, tab2 = st.tabs(["Decode me", "Team"])

with tab1:
    import base64
    @st.cache
    def load_image(path):
        with open(path, 'rb') as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        return encoded

    def image_tag(path):
        encoded = load_image(path)
        tag = f'<img src="data:image/png;base64,{encoded}">'
        return tag

    def background_image_style(path):
        encoded = load_image(path)
        style = f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        '''
        return style

    st.markdown(
        """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


    with st.sidebar:
        st.header("Decode the face")
        st.markdown('Upload a picture/screenshot of a person or people')
        uploaded_file = st.file_uploader("Choose a face image", type = ['png', 'jpg', 'jpeg'])

        recipe_button = st.button("Decode me")


    st.sidebar.image('face.jpg', use_column_width=True)

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
