# main.py
from streamlit_modules import default
from streamlit_modules import side_bar
from streamlit_modules import page_1
from streamlit_modules import page_2
from streamlit_modules import page_3

def main():
    # Set up page layout and theme
    default.default_page_config()

    # Show sidebar and get current page
    page, selected_model = side_bar.show_sidebar(['Model Testing', 'App', 'About'])

    # Display the selected page
    if page == "Model Testing":
        page_1.show_page(selected_model)
    elif page == "App":
        # you might need to add model selection for the App page as well
        page_2.show_page(None)
    elif page == "About":
        page_3.show_page()

if __name__ == "__main__":
    main()




