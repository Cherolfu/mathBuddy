import streamlit as st
from openai import OpenAI
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv

load_dotenv()


# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LATEX_PROMPT = """ REQUIREMENT:
Wrap latex with $. Example: $x^2$ for inline latex or
    $$
    ax^2 + bx + c
    $$
for new lines.
"""
IMAGE_READING_PROMPT = f"""
You are an math export, specializing in reading problem from an image. Return question from the image. {LATEX_PROMPT}
"""

HELPER_PROMPT = f"""
You are an math export, specializing in reading problem from an image, explain in steps and generate the answer.
 {LATEX_PROMPT}
"""

LEARNING_PROMPT = f"""
Please guide me by asking me questions step by step until I get the correct answer. For each question, please be simple and mention the knowledge point.
Start with the knowledge point and guide me towards the final solution please. Tell me if I am wrong and give me hints.
Please reevaluate each of my responses beforing responding.
 {LATEX_PROMPT}
"""


def encode_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


class MathHelper:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def _compose_content(self, image_base64, text_prompt):
        return [
            {"type": "text", "text": text_prompt},
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_base64}",
            },
        ]

    def query_vision(self, image, stream=True):
        image_base64 = encode_image_to_base64(image)
        content = self._compose_content(image_base64, IMAGE_READING_PROMPT)
        response_stream = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": content}],
            max_tokens=1000,
            stream=stream,
        )
        return response_stream

    def query(self, messages):
        stream = self.client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            stream=True,
        )
        return stream


def setup_streamlit_ui():
    st.title("Math Solver")
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image of a math problem", type=["png", "jpg", "jpeg"]
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "image" not in st.session_state:
        st.session_state.image = None
    if "learning_mode" not in st.session_state:
        st.session_state.learning_mode = None
    if "parse_question" not in st.session_state:
        st.session_state.parse_question = None

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption="Uploaded Math Problem", use_column_width=True)
        learning_mode = st.sidebar.toggle("Activate learning mode")

        st.session_state.image = image
        st.session_state.learning_mode = learning_mode
        return image
    return None


def show_question():
    if st.session_state.parse_question:
        with st.expander("Question from the image", expanded=True):
            st.write(st.session_state.parse_question)


def main():
    openai_helper = MathHelper(api_key=OPENAI_API_KEY)
    image = setup_streamlit_ui()

    show_question()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages[2:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.sidebar.button("Solve"):
        response = openai_helper.query_vision(image, stream=False)
        response = response.choices[0].message.content
        st.session_state.parse_question = response

        show_question()
        st.session_state.messages.append(
            {"role": "assistant", "content": st.session_state.parse_question}
        )
        text_prompt = (
            LEARNING_PROMPT if st.session_state.learning_mode else HELPER_PROMPT
        )

        st.session_state.messages.append({"role": "user", "content": text_prompt})

        stream = openai_helper.query(st.session_state.messages)
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

    if prompt := st.chat_input("Follow up question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = openai_helper.query(st.session_state.messages)
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
