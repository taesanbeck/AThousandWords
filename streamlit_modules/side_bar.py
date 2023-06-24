# sidebar.py
import streamlit as st
from PIL import Image

def show_sidebar(page_options):  
    st.sidebar.image('streamlit_modules/media/gmu_image.png', output_format='auto',channels='RGB')
    # Add navigation
    page = st.sidebar.selectbox("Go to", page_options, key="sidebar_selectbox")

    # Add a selectbox for CV model
    if page == "Model Testing":
        selected_model = st.sidebar.selectbox('Select CV Model', ['YOLO', 'DINO'])
    else:
        selected_model = None

    return page, selected_model

def show_sidebar_2():
    st.sidebar.title("Integrated Application")
    # Add a button
    button = st.sidebar.button('Run model')
    return button

def show_sidebar_3():
    st.sidebar.title("About")

    
