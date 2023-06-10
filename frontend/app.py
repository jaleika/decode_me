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
import cv2
import numpy as np

# from decode.ml_logic.visual_representation.main import visual  ##added

import time

url = "http://127.0.0.1:8000"

# Use the full page instead of a narrow central column
st.set_page_config(
    page_title="Decode me",
    page_icon="☝",
    layout="wide",
    initial_sidebar_state="expanded",
)

tab1, tab2 = st.tabs(["Decode me", "Team"])

with tab1:

    @st.cache_data
    def visual(img, res):
        detection = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise",
        }
        corners = res["corners"]
        mood = res["mood"]
        print(mood)

        scale_factor = 900 / img.shape[1]
        new_dim = (int(scale_factor * img.shape[1]), int(scale_factor * img.shape[0]))
        img_900 = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
        print(image.shape)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for index, face_corners in enumerate(corners[0]):
            top_left = face_corners[0:2]
            top_left2 = [face_corners[0], face_corners[1] - 14]
            right_bottom = face_corners[2:]
            if mood[index] == "angry":
                color = (255, 0, 0)
            if mood[index] == "disgust":
                color = (0, 255, 0)
            if mood[index] == "fear":
                color = (0, 0, 0)
            if mood[index] == "happy":
                color = (7, 223, 255)
            if mood[index] == "neutral":
                color = (255, 255, 255)
            if mood[index] == "sad":
                color = (111, 111, 111)
            if mood[index] == "surprise":
                color = (255, 128, 0)
            cv2.rectangle(img2, top_left, right_bottom, color, 4)
            cv2.putText(
                img2, mood[index], top_left2, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1
            )
        return img2

    imageLocation = st.empty()
    imageLocation.image("face.jpg", width=900)
    #    st.image('face.jpg', width = 900)

    st.markdown(
        """
    <style>
      [data-testid=stSidebar] {
        background-color: #5B5B5B;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Decode the face")
        st.markdown("Upload a picture of a person or people")
        uploaded_image = st.file_uploader(
            "Choose a face image", type=["png", "jpg"], accept_multiple_files=False
        )
        if uploaded_image:
            img_bytes = uploaded_image.getvalue()
            imageLocation.image(img_bytes, width=900)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if st.button(f"Get result"):

                # url_endpoint = f"{url}predict"
                img_bytes = uploaded_image.getvalue()
                # st.image(Image.open(uploaded_image))
                res = requests.post(url=url + "/predict", files={"image": img_bytes})
                result_final = visual(image, res.json())  ##added
                # imageLocation.image(result_final, width=900)  ##added
                st.write(f"The mood of the picture is: {res.json()['mood']}")
                # TODO: change img_bytes in the following so that it contains colored boxes around the faces encoding the mood the emotion as text

                imageLocation.image(result_final, width=900)

    st.sidebar.image("face4.jpg", use_column_width=True)

with tab2:
    col1, col2, col3 = st.columns((4, 4, 2))

    with col1:
        st.markdown("## Meet the Decode me Team")
        st.markdown("#### Natasha 🍉")
        # st.image('Mariia.png', width = 250)
        st.markdown("#")
        st.markdown("#### Isabella 🍭")
        # st.image('Lisa1.png', width = 250)

    with col2:
        st.markdown("#")
        st.markdown("#")
        st.markdown("#### Hayri 🎈")
        # st.image('Malory.png', width = 250)
        st.markdown("#")
        st.markdown("#### Isabel 🍹")
        # st.image('David1.png', width = 250)
