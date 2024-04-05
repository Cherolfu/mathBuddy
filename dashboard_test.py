import streamlit as st
from openai import OpenAI
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
import requests
import re

load_dotenv()


# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_ID = os.getenv("WOLFRAM_ALPHA_APP_ID")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LATEX_PROMPT = """ REQUIREMENT:
1. Wrap LaTeX code with single dollar signs for inline LaTeX. For example: $x^2$ renders inline as \(x^2\).
2. Use double dollar signs for equations that should appear centered on their own lines. For example:
    $$
    ax^2 + bx + c
    $$
renders as a centered equation \(ax^2 + bx + c\).

3. Use the \dfrac{}{} command for display fractions, which renders fractions on separate lines. Here's how you can apply it:
    $$
    \dfrac{numerator}{denominator}
    $$
Replace 'numerator' and 'denominator' with the appropriate expressions.

4. To correctly denote exponentiation, always use the caret (^) symbol. For instance, $2^x$ indicates \(2\) raised to the power of \(x\), distinctly different from $2x$, which represents \(2\) multiplied by \(x\).

5. When writing exponentiation with more than one term in the exponent, enclose the terms in curly braces. For example, $2^{x+1}$ renders as \(2^{x+1}\), which is correct.

6. Be attentive to the placement of curly braces {} to ensure grouping of terms, especially in exponents and fractions.

EXAMPLE:
The equation
$$
\dfrac{66-2^x}{2^x+3} = \dfrac{4-2^x}{2^{x+1}+6}
$$
should be input into LaTeX as shown to ensure correct rendering and interpretation by LaTeX compilers and mathematical software.

Remember: Consistency and attention to detail in formatting are key to correctly rendering mathematical expressions in LaTeX.
"""

IMAGE_READING_PROMPT = f"""
You are an math export, specializing in reading problem from an image. Return question from the image concisely, and please only show the exact content that shows in the image. {LATEX_PROMPT}
"""

HELPER_PROMPT = f"""
You are a math expert, specializing in reading problems from an image.

If the problem has choices to select.
First, please present the final answer within a single rectangular box. Then, provide a step-by-step explanation as concise as possible.

If the problem does NOT have choices to select.
First, provide a step-by-step explanation as concise as possible. Then, present the final answer within a single rectangular box.
 {LATEX_PROMPT}
"""

LEARNING_PROMPT = f"""
First, based on the problem, please provide 2-3 knowledge points using concise language with the bold subtitle "Knowledge Points". Avoid considering “Simplify the expression” and “Combining terms” as standalone knowledge points.

Then, having another bold subtitle "Now, let's work through the problem together with a few step-by-step guiding questions." guide me with asking one concise, guiding question in the format of multiple choice (4 different choices) toward the correct solution.

Once I answered each guiding question, please tell me know the correctness. If it's correct, please proceed to the next guiding question. If it’s wrong or the user says “I don’t know”, provide more hints instead of directly telling me the correct answer.
{LATEX_PROMPT}
"""

def encode_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


class MathHelper:
    def __init__(self, api_key, wolf_api_key):
        self.client = OpenAI(api_key=api_key)
        self.wolf_api_key = wolf_api_key

    def _compose_content(self, image_base64, text_prompt):
        return [
            {"type": "text", "text": text_prompt},
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_base64}",
            },
        ]

    def query_wolfram_alpha(self, query):
        url = "http://api.wolframalpha.com/v2/query"
        params = {
            "input": query,
            "appid": self.wolf_api_key,
            "output": "JSON",
        }
        response = requests.get(url, params=params)
        response_data = response.json()
        #print(query)
        #print(response_data)
        return response_data

        '''
        try:
            for pod in response_data['queryresult']['pods']:
                if pod['title'] == 'Real solution':
                # Assuming there's only one subpod under 'Result' for simplicity
                    result_text = pod['subpods'][0]['plaintext']
                    return result_text
        except KeyError as e:
        # Print and return the KeyError if it occurs
            error_message = f"KeyError: The key {e} is missing in the response."
            print(error_message)
            return error_message
        except IndexError as e:
            # Print and return the IndexError if it occurs
            error_message = f"IndexError: The index is out of range in the response."
            print(error_message)
            return error_message
        '''

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


def extract_equation(message):
    # Regular expression
    pattern = r'\$\$(.*?)\$\$'
    matches = re.findall(pattern, message, re.DOTALL)

    # Check if we found any matches
    if matches:
        equation = "solve for " + matches[0].strip()
        return equation
    else:
        return "No equation found in the message."

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
    openai_helper = MathHelper(api_key=OPENAI_API_KEY, wolf_api_key=APP_ID)
    image = setup_streamlit_ui()

    show_question()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages[2:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.sidebar.button("Solve"):
        response = openai_helper.query_vision(image, stream=False)
        #print("xxxxxxx1")
        #print(response)
        response = response.choices[0].message.content
        #print("xxxxxxx2")
        #print(response)
        #content_type = type(response)
        #print(content_type)
        query_prompt = response
        st.session_state.parse_question = response
        #response = openai_helper.query_wolfram_alpha(response)

        extracted_equation = extract_equation(query_prompt)
        print(extracted_equation)
        formatted_query_prompt = f"'{extracted_equation}'"
        print(formatted_query_prompt)

        show_question()
        st.session_state.messages.append(
            {"role": "assistant", "content": st.session_state.parse_question}
        )
        text_prompt = (
            LEARNING_PROMPT if st.session_state.learning_mode else HELPER_PROMPT
        )

        final_result = openai_helper.query_wolfram_alpha(formatted_query_prompt)
        Learning_Wolf_Prompt = f"Please learn the following JSON output from Wolfram Alpha: '{final_result}'"
        st.session_state.messages.append({"role": "user", "content": Learning_Wolf_Prompt})

        st.session_state.messages.append({"role": "user", "content": text_prompt})

        stream = openai_helper.query(st.session_state.messages)
        #print("xxxxxxx3")
        #print(stream)
        #final_result = openai_helper.query_wolfram_alpha(stream)
        with st.chat_message("assistant"):
            #final_result = openai_helper.query_wolfram_alpha(formatted_query_prompt)
            #response = st.write(final_result)
            #print(response)
            response = st.write_stream(stream)
            #print("####response")
            #print(response)
            #print(st.session_state.messages)
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