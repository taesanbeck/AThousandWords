from streamlit_modules import default
from streamlit_modules import side_bar
from streamlit_modules import page_1
from streamlit_modules import page_2
from streamlit_modules import page_3

def main():
    # Set up page layout and theme
    default.default_page_config()

    # Show sidebar and get current page
    page = side_bar.show_sidebar(['Page 1', 'Page 2', 'Page 3'])

    # Display the selected page
    if page == "Page 1":
        button, slider, selectbox = side_bar.show_sidebar_1()
        page_1.show_page(button, slider, selectbox)
    elif page == "Page 2":
        slider, selectbox = side_bar.show_sidebar_2()
        page_2.show_page(slider, selectbox)
    elif page == "Page 3":
        button = side_bar.show_sidebar_3()
        page_3.show_page(button)

if __name__ == "__main__":
    main()



