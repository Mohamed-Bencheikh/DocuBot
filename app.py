import streamlit as st
import os
import tempfile
import time
from utilities import display_pdf, get_response, get_pdf_text, summarize

def main():
    st.set_page_config(page_title="PDF Viewer", layout="wide", page_icon='./logo.png')
    st.title(":blue[DocuBot]",anchor=False)
    st.write("View and chat with your PDF")

    if 'messages' not in st.session_state:
        st.session_state.messages = [{'role': 'assistant', "content": "Hello! Upload a document and let's get started."}]
    state = True
    uploaded_file = st.sidebar.file_uploader("Upload your PDF File", type="pdf")
    if uploaded_file:
        state = False
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())  # Write the PDF content
            pdf_text = get_pdf_text(file_path)
            pdf_frame = display_pdf(file_path)
            st.sidebar.markdown(pdf_frame, unsafe_allow_html=True)
    user_prompt = st.chat_input("What do you wanna know about the document?", disabled=state)
    if user_prompt:
        st.session_state.messages.append({'role': 'user', "content": user_prompt})
        with st.spinner("..."):
            response = get_response(pdf_text, user_prompt)
            time.sleep(2)
            st.session_state.messages.append({'role': 'assistant', "content": response})

    if st.sidebar.button(label="summarize"):
        st.session_state.messages.append({'role': 'user', "content": "Summarize the document"})
        with st.spinner("..."):
            summary = summarize(pdf_text, max_length=200)
            st.session_state.messages.append({'role': 'assistant', "content": "Summary:  <br>"+summary})

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'], unsafe_allow_html=True)            

if __name__ == "__main__":
    main()
