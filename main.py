#Import to get the helper functions place in the text_storage file
import text_storage

#Import Streamlit and the openai API
import streamlit as st
from openai import OpenAI

#libraries for displaying images
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO

#Generate unique keys for the 4 display objects
button1_key = "button1"
button2_key = "button2"
button3_key = "button3"
input1_key = "input1"


# Set OpenAI API key from Streamlit secrets file in the path ".streamlit/secrets.toml"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
gpt_model = "gpt-4"#"gpt-3.5-turbo"


def first_response_generation(prompt):
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
        )
        response_text = st.write_stream(stream)
        with st.spinner("Generating image"):
            response_image = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response_image.data[0].url
            image = Image.open(BytesIO(requests.get(image_url).content))
            st.image(image)
    text_summary = text_storage.summarize_text(st.session_state.text_summary, response_text, gpt_model)
    st.session_state.text_summary = text_summary
    st.session_state.messages.append({"role": "assistant", "content": response_text, "image": image_url})

# Streamlit application
def iterative_response_generation(prompt):
    # Display assistant response in chat message container
    check = st.session_state.text_summary
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that describe the scenary of a bedtime story."},
                {"role":"user","content": "Summarize the previous fairytale for further extending the story"},
                {"role":"assistant","content": st.session_state.text_summary},
                {"role": "user", "content": "While taking the summarized fairytale into account; create and  describe a scenary in less than 150 words for the following prompt: "+prompt}
            ],
            stream=True,
        )
        response_text = st.write_stream(stream)

        image_prompt = "Create an image for the following context:" + text_storage.summarize_text_for_image(st.session_state["text_summary"],response_text,3,1)
        with st.spinner("Generating image"):
            response_image = client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response_image.data[0].url
            image = Image.open(BytesIO(requests.get(image_url).content))
            st.image(image)

        st.write("Choose an option on how to extend the Bedtime Story")
        option_1 = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
                {"role": "user", "content": "Can you please summarize the previous bedtime story. Later use this summary to extend the Bedtime Story."},
                {"role": "assistant", "content": st.session_state.text_summary},
                {"role": "user",
                 "content": "Create and just describe ascenary in less than 150 words for the following prompt, while taking the summary into account: " + prompt},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": "Create three key words just seperated by commas to extend the story, with the following input: " + prompt}
            ],
            max_tokens=9
        )


        option_2 = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
                {"role": "user", "content": "Can you please summarize the previous bedtime story. Later use this summary to extend the Bedtime Story."},
                {"role": "assistant", "content": st.session_state.text_summary},
                {"role": "user",
                 "content": "Create and just describe ascenary in less than 150 words for the following prompt, while taking the summary into account: " + prompt},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": "Create three key words just seperated by commas to extend the story, with the following input: " + prompt}
            ],
            max_tokens=9
        )

        option_3 = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
                {"role": "user", "content": "Can you please summarize the previous bedtime story. Later use this summary to extend the Bedtime Story."},
                {"role": "assistant", "content": st.session_state.text_summary},
                {"role": "user",
                 "content": "Create and just describe ascenary in less than 150 words for the following prompt, while taking the summary into account: " + prompt},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": "Create three key words just seperated by commas to extend the story, with the following input: " + prompt}
            ],
            max_tokens=9
        )
        st.session_state.messages.append({"role": "assistant", "content": response_text, "image": image_url})

        st.write("Option 1")
        option_1 = option_1.choices[0].message.content
        with st.spinner("Creating the story"):
            st.button(option_1,key=button1_key, on_click=create_option_content,args=[prompt,option_1,response_text])

        st.write("Option 2")
        option_2 = option_2.choices[0].message.content
        with st.spinner("Creating the story"):
            st.button(option_2, key=button2_key, on_click=create_option_content,args=[prompt,option_2,response_text])

        st.write("Option 3")
        option_3 = option_3.choices[0].message.content
        with st.spinner("Creating the story"):
            st.button(option_3, key=button3_key, on_click=create_option_content,args=[prompt,option_3,response_text])



def create_option_content(prompt,keywords,response_text):
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that describe the scenary of a bedtime story."},
                {"role": "user", "content": "Summarize the previous fairytale"},
                {"role": "assistant", "content": st.session_state.text_summary},
                {"role": "user",
                 "content": "Create and just describe ascenary in less than 150 words for the following prompt, while taking the summary into account: " + prompt},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content":"Create three key words just seperated by commas to extend the story, with the following input: " + prompt},
                {"role": "assistant", "content": keywords},
                {"role": "user", "content": "Creatively extend the story from the fairytale summary while taking these keyword into acount: "+keywords+"."}
            ],
            stream=True,
        )
        response_text = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    text_summary= text_storage.summarize_text(st.session_state.text_summary,response_text,gpt_model)
    st.session_state.text_summary=text_summary


def create_final_story():
    with st.chat_message("assistant"):
        #print(st.session_state.text_summary)
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
                {"role": "user", "content": "Create a creative bedtime story for a child while using the following summary: "+st.session_state.text_summary}
            ],
            stream=True,
        )
        st.markdown("### Final Story")
        response_text = st.write_stream(stream)
        image_prompt = "Create an image for the following context:" + response_text
        with st.spinner("Generating image"):
            response_image = client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response_image.data[0].url
            image = Image.open(BytesIO(requests.get(image_url).content))
            st.image(image)
    st.session_state.messages.append({"role": "assistant", "content": response_text, "image": image_url})

    text_summary= text_storage.summarize_text(st.session_state.text_summary,response_text,gpt_model)
    st.session_state.text_summary = text_summary

def main():
    #first_iteration = True
    st.title("Fairytale Generator")

    # Set a default gpt_model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = gpt_model

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    #Initialize text summary
    if "text_summary" not in st.session_state:
        st.session_state.text_summary = ''

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message:
                try:
                    image = Image.open(BytesIO(requests.get(message["image"]).content))
                except UnidentifiedImageError or UnboundLocalError:
                    st.write("Image Couldn't be loaded")
                st.image(image)
            if "option" in message:
                st.markdown("The choosen extension: \n"+message["option"])

    st.sidebar.markdown("***Hint***: \nTo complete the story please enter: 'Complete the story'")
    # React to user input
    if prompt := st.chat_input("Feed me with ideas for a Bedtime story... ", key=input1_key):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        if st.session_state.messages[-1]["role"] != "assistant":
            if 'complete the story' not in prompt.lower():
                if len(st.session_state.text_summary) ==0:
                    first_response_generation(prompt)
                    #first_iteration = False
                else:
                    iterative_response_generation(prompt)
            else:
                if len(st.session_state.text_summary) == 0:
                    with st.chat_message("assistant"):
                        st.write("Give me first ideas to generate a story")
                        st.session_state.messages.append({"role": "assistant", "content": "Give me first ideas to generate a story"})
                else:
                    create_final_story()
    except IndexError as e:
        st.write("Start your Story")

    #if len(st.session_state.messages) ==1:




if __name__ == "__main__":
    main()
    #Explain the scenery of a fairytale castle on a dark and stormy night.
    #A knight is riding slowly towards the castle
