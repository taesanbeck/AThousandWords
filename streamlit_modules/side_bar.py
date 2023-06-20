# sidebar.py
import streamlit as st
from PIL import Image


def show_sidebar(page_options):
    st.sidebar.image('streamlit_modules/media/gmu_image.png', output_format='auto',channels='RGB')
    # Add navigation
    page = st.sidebar.selectbox("Go to", page_options, key="sidebar_selectbox")

    return page

def show_sidebar_1():   
    # Add a slider
    slider = st.sidebar.slider('Model Confidence Level Slider', min_value=0, max_value=100, value=50)
    # Add a selectbox
    selectbox = st.sidebar.selectbox('Select Model', ['Yolo', 'DINO', 'DINOv2'])
    return slider, selectbox

def show_sidebar_2():
    st.sidebar.title("Thousand Words - Page 2")
    # Add a slider
    slider = st.sidebar.slider('Model Slider', min_value=0, max_value=100, value=50)
    # Add a selectbox
    selectbox = st.sidebar.selectbox('Select Me', ['Option 1', 'Option 2', 'Option 3'])
    return slider, selectbox

def show_sidebar_3():
    st.sidebar.title("About")
    # Add a button
    button = st.sidebar.button('Button 1')
    return button
