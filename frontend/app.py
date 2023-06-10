import streamlit as st
import requests
from time import sleep
from stqdm import stqdm

##local url
#url = 'http://localhost:8000/'
url = 'https://decode-eykha3qtfq-ew.a.run.app'

#Use the full page instead of a narrow central column
st.set_page_config(page_title='Decode me',
                page_icon = '☝',
                layout = 'wide',
                initial_sidebar_state = 'expanded')

tab1, tab2, tab3 = st.tabs(["Decode others","Decode me", "Team"])

with tab1:
    c = st.empty()


    st.markdown("""
    <style>
      [data-testid=stSidebar] {
        background-color: #5B5B5B;
    }
    </style>
    """,unsafe_allow_html=True)


    with st.sidebar:
        st.header("Picture decoding")
        st.markdown('Upload a picture.')
        uploaded_image = st.file_uploader("Choose a face image", type = ['png', 'jpg'], accept_multiple_files=False)
        if uploaded_image:
            img_bytes = uploaded_image.getvalue()
            c.image(img_bytes, width = 900)
            if st.button(f'Get result'):

                img_bytes = uploaded_image.getvalue()
                res = requests.post(url = url + "/predict", files = {'image': img_bytes})

                st.write(f"The mood of the picture is: {res.json()['mood']}")
                # TODO: change img_bytes in the following so that it contains colored boxes around the faces encoding the mood the emotion as text

                c.image(img_bytes, width = 900)

    c.image('face.jpg', width = 900)

    st.sidebar.image('face4.jpg', use_column_width=True)

with tab2:
        #####display a widget that returns pictures from th users's webcam.
    #if st.button('Decode me :sunglasses:', use_container_width=True):
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
    # To read image file buffer as bytes:
        bytes_data = img_file_buffer.getvalue()
        res = requests.post(url = url + "/predict", files = {'image': bytes_data})

        st.write(f"Your mood looks like: {res.json()['mood']}")


with tab3:
    col1, col2, col3 = st.columns((4, 4, 2))

    with col1:
        st.markdown('## Meet the Team')
        st.markdown('#### Natasha 🍉')
        st.image('Natasha.jpg', width = 250)
        st.markdown('#')
        st.markdown('#### Isabella 🍭')
        #st.image('Lisa1.png', width = 250)

    with col2:
        st.markdown('#')
        st.markdown('#')
        st.markdown('#### Hayri 🎈')
        st.image('Hayri.jpeg', width = 250)
        st.markdown('#')
        st.markdown('#### Isabel 🍹')
        st.image('Isabel.jpg', width = 250)
