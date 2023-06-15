import streamlit as st
import requests
import numpy as np
import cv2


url = 'http://127.0.0.1:8000'
##local url
#url = 'http://localhost:8000/'
#url = " https://decode-eykha3qtfq-ew.a.run.app"

# Use the full page instead of a narrow central column
st.set_page_config(
    page_title="Decode me",
    page_icon="☝",
    layout="wide",
    initial_sidebar_state="expanded",
)

tab1, tab2, tab3 = st.tabs(["Decode others","Decode me", "Team"])

with tab1:
    c = st.empty()

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
        img2 = cv2.cvtColor(img_900, cv2.COLOR_BGR2RGB)
        for index, face_corners in enumerate(corners[0]):
            top_left = (
                int(scale_factor * face_corners[0]),
                int(scale_factor * face_corners[1]),
            )
            top_left2 = [
                int(scale_factor * face_corners[0]),
                int(
                    scale_factor * face_corners[1]
                    - int(scale_factor * (face_corners[2] - face_corners[0]) / 7)
                ),
            ]
            right_bottom = (
                int(scale_factor * face_corners[2]),
                int(scale_factor * face_corners[3]),
            )
            color_dict = {
                "angry": (255, 0, 0),
                "disgust": (0, 255, 0),
                "fear": (0, 0, 0),
                "happy": (7, 223, 255),
                "neutral": (255, 255, 255),
                "sad": (111, 111, 111),
                "surprise": (255, 128, 0),
            }
            color = color_dict.get(
                mood[index], (255, 255, 255)
            )  # If no match, default to white
            cv2.rectangle(
                img2,
                top_left,
                right_bottom,
                color,#(254, 52, 162),#
                int(scale_factor * (face_corners[2] - face_corners[0]) / 50),#3,#
            )
            """cv2.putText(
                img2,
                mood[index],
                tuple(top_left2),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale_factor * (face_corners[2] - face_corners[0]) / 100,
                color,
                int(scale_factor * (face_corners[2] - face_corners[0]) / 50),
                lineType=cv2.LINE_AA,
            )"""
        return img2

    imageLocation = st.empty()
    imageLocation.image("face.jpg", width=900)

    # if st.button

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

    #####display a widget that returns pictures from th users's webcam.
    # img_file_buffer = st.camera_input("Take a picture")

    # if img_file_buffer is not None:
    #     # To read image file buffer as bytes:
    #     bytes_data = img_file_buffer.getvalue()
    #     # Check the type of bytes_data:
    #     # Should output: <class 'bytes'>
    #     st.write(type(bytes_data))

    with st.sidebar:
        st.header("Decode the face")
        st.markdown("Upload a picture.")
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
                res = requests.post(
                    url=url + "/predict", files={"image": img_bytes}, timeout=300
                )
                result_final = visual(image, res.json())

                st.write(f"The mood of the picture is: {res.json()['mood']}")
                # TODO: change img_bytes in the following so that it contains colored boxes around the faces encoding the mood the emotion as text

                imageLocation.image(result_final, width=900)

    st.sidebar.image("face4.jpg", use_column_width=True)

with tab2:
    location = st.empty()
    img_file_buffer = location.camera_input("Take a picture")
    if img_file_buffer is not None:
    # To read image file buffer as bytes:
        bytes_data = img_file_buffer.getvalue()
        res = requests.post(url = url + "/predict", files = {'image': bytes_data})
        ########
        ### this image needs to get a rectangle and a text:
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        location.image(cv2_img)
        st.write(f"Your mood looks like: {res.json()['mood']}")


with tab3:
    col1, col2, col3 = st.columns((4, 4, 2))

    with col1:
        st.markdown('## Meet the Team')
        st.markdown('#### Natasha 🍉')
        st.image('Natasha.jpg', width = 250)
        st.markdown('#')
        st.markdown('#### Isabella 🍭')
        st.image('Photo_Isabella_quad.jpeg', width = 250)

    with col2:
        st.markdown('#')
        st.markdown('#')
        st.markdown('#### Hayri 🎈')
        st.image('Hayri.jpeg', width = 250)
        st.markdown('#')
        st.markdown('#### Isabel 🍹')
        st.image('Isabel.jpg', width = 250)
