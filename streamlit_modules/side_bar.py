# sidebar.py
import streamlit as st
from PIL import Image


def show_sidebar(page_options):
    st.sidebar.image('streamlit_modules/media/gmu_image.png', output_format='auto',channels='RGB')
    # Add navigation
    page = st.sidebar.selectbox("Go to", page_options, key="sidebar_selectbox")

    return page

def show_sidebar_1():   
    # Add a selectbox
    selectbox = st.sidebar.selectbox('Select CV Model', ['Yolo', 'DINO'])
    return selectbox

def show_sidebar_2():
    st.sidebar.title("Integrated Application")
    # Add a button
    button = st.sidebar.button('Run model')
    return button

def show_sidebar_3():
    st.sidebar.title("About")
    # Add a button
    button = st.sidebar.button('Button 1')
    return button
