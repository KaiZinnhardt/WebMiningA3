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
gpt_model = "gpt-3.5-turbo" #"gpt-4"


def moderation_check(input_message):
    input_message = str(input_message)
    list_message = [input_message[i:i+2000] for i in range(0, len(input_message), 2000)]
    for message in list_message:
        response = client.moderations.create(input=message)
        if response.results[0].flagged == True:
            return True
    return False



def first_response_generation(prompt):
    """
    This the response to the first prompt that the user has entered
    :param prompt: the user input for which the bot bot will generate text
    """
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        #API call to generate the response
        message = [
                {"role": "system",
                 "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
                {"role": "user", "content": "Describe the context for a bedtime story using the following prompt:"+prompt+"\n Add a cliffhanger at the end of the story."}
            ]
        allowed_tokens = text_storage.max_token_calculator(message,gpt_model)
        mod_bool = moderation_check(message)
        if mod_bool == True:
            st.write("The input is senitized please use an appropriate prompt")
            return None
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=message,
            stream=True,
            max_tokens=allowed_tokens
        )
        #create the response text as a stream
        response_text = st.write_stream(stream)
        image_prompt = "Create an image in a cartoon like style for the following context:" + text_storage.summarize_text_for_image('', response_text,5, 2)
        mod_bool=moderation_check(image_prompt)
        if mod_bool == True:
            st.write("The input is senitized please use an appropriate prompt")
            return None
        #Create a spinner to create a visual while waiting for the response
        with st.spinner("Generating image"):
            #API call to generate the image
            response_image = client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            #extract the url
            image_url = response_image.data[0].url
            try:
                #display the image
                image = Image.open(BytesIO(requests.get(image_url).content))
                st.image(image)
            except UnidentifiedImageError or UnboundLocalError: #Catch potential error messages that could occur.
                st.markdown("***Image Couldn't be loaded***")
    #Create a summary of the text, where 'st.session_state.text_summary' is ''
    text_summary = text_storage.summarize_text(st.session_state.text_summary, response_text, gpt_model)
    #Save the summary in a streamlit session state variable
    st.session_state.text_summary = text_summary
    #Append the text messages into the streamlit session state variable
    st.session_state.messages.append({"role": "assistant", "content": response_text, "image": image_url})

def iterative_response_generation(prompt):
    """
    This the iterative response once the user has started the conversation. Thus this includes the three potential options
    :param prompt: the user input for which the bot will generate text
    """
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # API call to generate the response, include the summary of the previous responses to give the API the necessary knowledge in a cost efficient way
        message=[
                {"role": "system",
                 "content": "You are a helpful assistant that describe the scenary of a bedtime story."},
                {"role":"user","content": "Summarize the previous fairytale for further extending the story"},
                {"role":"assistant","content": st.session_state.text_summary},
                {"role": "user", "content": "While taking the summarized fairytale into account; create and  describe a scenary in less than 150 words for the following prompt: "+prompt}
            ]
        allowed_tokens = text_storage.max_token_calculator(message, gpt_model)
        print(allowed_tokens)
        mod_bool = moderation_check(message)
        if mod_bool == True:
            st.write("The input is senitized please use an appropriate prompt")
            return None
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=message,
            stream=True,
            max_tokens=allowed_tokens
        )
        #create the response text as a stream
        response_text = st.write_stream(stream)
        #creates the prompt of the image. Thereby, we use a summary of the response to create a cost efficient dialogue
        image_prompt = "Create an image in a cartoon like style for the following context:" + text_storage.summarize_text_for_image(st.session_state["text_summary"],response_text,5,2)
        #Create an interactive graphic while waiting for the image to be generated
        mod_bool = moderation_check(image_prompt)
        if mod_bool == True:
            st.write("The input is senitized please use an appropriate prompt")
            return None
        with st.spinner("Generating image"):
            # API call to generate the image
            response_image = client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            # extract the url
            image_url = response_image.data[0].url
            try:
                # display the image
                image = Image.open(BytesIO(requests.get(image_url).content))
                st.image(image)
            except UnidentifiedImageError or UnboundLocalError: #Catch potential error messages that could occur.
                st.markdown("***Image Couldn't be loaded***")
        #Saves the first part of the interaction
        st.session_state.messages.append({"role": "assistant", "content": response_text, "image": image_url})
        #Create an informative comment for the user
        st.write("Choose an option on how to extend the Bedtime Story")
        #Create the keywords for the three options
        #API call for the first option
        option_message = [
                {"role": "system",
                 "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
                {"role": "user", "content": "Can you please summarize the previous bedtime story. Later use this summary to extend the Bedtime Story."},
                {"role": "assistant", "content": st.session_state.text_summary},
                {"role": "user",
                 "content": "Create and just describe ascenary in less than 150 words for the following prompt, while taking the summary into account: " + prompt},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": "Create three key words just seperated by commas to extend the story, with the following input: " + prompt}
            ]
        if text_storage.max_token_calculator(message,gpt_model)<9:
            option_message =[
                {"role": "system",
                 "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
                {"role": "user", "content": "Can you please summarize the previous bedtime story. Later use this summary to extend the Bedtime Story."},
                {"role": "assistant", "content": st.session_state.text_summary},
                {"role": "user",
                 "content": "Create and just describe ascenary in less than 150 words for the following prompt, while taking the summary into account: " + prompt}]
        mod_bool = moderation_check(option_message)
        if mod_bool == True:
            st.write("The input is senitized please use an appropriate prompt")
            return None
        option_1 = client.chat.completions.create(
            model=st.session_state["openai_model"],
            #gives as much contentext as possible while staying cost efficient
            messages= option_message,
            #limits the output space such that only 1-3 keywords are returned
            max_tokens=9
        )
        # API call for the second option
        option_2 = client.chat.completions.create(
            model=st.session_state["openai_model"],
            # gives as much contentext as possible while staying cost efficient
            messages=option_message,
            # limits the output space such that only 1-3 keywords are returned
            max_tokens=9
        )
        # API call for the third option
        option_3 = client.chat.completions.create(
            model=st.session_state["openai_model"],
            # gives as much contentext as possible while staying cost efficient
            messages=option_message,
            # limits the output space such that only 1-3 keywords are returned
            max_tokens=9
        )
        #Provides the user with the three options as buttons with a text above showing the number of option it displays
        st.write("Option 1")
        option_1 = option_1.choices[0].message.content
        st.button(option_1,key=button1_key, on_click=create_option_content,args=[prompt,option_1,response_text])

        st.write("Option 2")
        option_2 = option_2.choices[0].message.content
        st.button(option_2, key=button2_key, on_click=create_option_content,args=[prompt,option_2,response_text])

        st.write("Option 3")
        option_3 = option_3.choices[0].message.content
        st.button(option_3, key=button3_key, on_click=create_option_content,args=[prompt,option_3,response_text])



def create_option_content(prompt,keywords,response_text):
    """

    Creates the response to the keywords that the user choose.
    :param prompt: the input form the user
    :param keywords: the keywords that the user choose
    :param response_text: the previous response text of the first part of the interaction
    """
    #The context where the message will be given in.
    with st.chat_message("assistant"):
        # API call to generate the response, include the summary of the previous responses, the response text of the first part of this response, the three key words to give the API the necessary knowledge in a cost efficient way
        message = [
                {"role": "system",
                 "content": "You are a helpful assistant that describe the scenary of a bedtime story."},
                {"role": "user", "content": "Summarize the previous fairytale"},
                {"role": "assistant", "content": st.session_state.text_summary},
                {"role": "user",
                 "content": "Create and just describe ascenary in less than 150 words for the following prompt, while taking the summary into account: " + prompt},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content":"Create three key words just seperated by commas to extend the story, with the following input: " + prompt},
                {"role": "assistant", "content": keywords},
                {"role": "user", "content": "Creatively extend the story from the fairytale summary while taking the keyword of "+keywords+" into acount. Add a cliff hanger at the end of the story"}
            ]
        allowed_tokens = text_storage.max_token_calculator(message,gpt_model)
        mod_bool = moderation_check(message)
        if mod_bool == True:
            st.write("The input is senitized please use an appropriate prompt")
            return None
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=message,
            stream=True,
            max_tokens=allowed_tokens
        )
        # create the response text as a stream
        response_text = st.write_stream(stream)
    # Append the text messages into the streamlit session state variable
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    # Create a summary of the text
    text_summary= text_storage.summarize_text(st.session_state.text_summary,response_text,gpt_model)
    # Save the summary in a streamlit session state variable
    st.session_state.text_summary=text_summary


def create_final_story():
    """
    Creates the final story for the bedtime story while taking all of the information into account that it was given.
    """
    # The context where the message will be given in.
    print(st.session_state.text_summary)
    message = [
        {"role": "system",
         "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
        {"role": "user",
         "content": "Create a elaborate creative bedtime story, with a maximum of 1200 words, for a child while using the following summary: " + st.session_state.text_summary}
    ]
    allowed_tokens = text_storage.max_token_calculator(message,gpt_model)
    mod_bool = moderation_check(message)
    if mod_bool == True:
        st.write("The input is senitized please use an appropriate prompt")
        return None
    with st.chat_message("assistant"):
        # API call to generate the response, include the summary of the previous responses to give the API the necessary knowledge in a cost efficient way
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=message,
            stream=True,
            max_tokens=allowed_tokens
        )
        #Header of this section
        st.markdown("### Final Story")
        # create the response text as a stream
        response_text = st.write_stream(stream)
        # creates the prompt of the image. Thereby, we use a summary of the response to create a cost efficient dialogue
        image_prompt = "Create an image in a cartoon like style for the following context:" + text_storage.summarize_text_for_image('',response_text,5,2)
        # Create an interactive graphic while waiting for the image to be generated
        with st.spinner("Generating image"):
            # API call to generate the image
            response_image = client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            # extract the url
            image_url = response_image.data[0].url
            try:
                # display the image
                image = Image.open(BytesIO(requests.get(image_url).content))
                st.image(image)
            except UnidentifiedImageError or UnboundLocalError: #Catch potential error messages that could occur.
                st.markdown("***Image Couldn't be loaded***")
    # Append the text messages into the streamlit session state variable
    st.session_state.messages.append({"role": "assistant", "content": response_text, "image": image_url})
    # Create a summary of the text
    text_summary= text_storage.summarize_text(st.session_state.text_summary,response_text,gpt_model)
    # Save the summary in a streamlit session state variable
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
            #tries to fetch an image in the message state
            if "image" in message:
                try:
                    #displays the image
                    image = Image.open(BytesIO(requests.get(message["image"]).content))
                    st.image(image)
                except UnidentifiedImageError or UnboundLocalError: #Catch potential error messages that could occur.
                    st.markdown("***Image Couldn't be loaded***")
    #Create the hint for the user how to complete the story
    st.sidebar.markdown("***Hint***: \nTo complete the story please enter: 'Complete the story'")
    # React to user input
    if prompt := st.chat_input("Feed me with ideas for a Bedtime story... ", key=input1_key):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    #Checks if the last message was the assistant or the user as the assistant should only come after the user
    try:
        if st.session_state.messages[-1]["role"] != "assistant":
            #Checks if the user wants to complete the story
            if 'complete the story' not in prompt.lower():
                #Decides on whether it is the first conversation on an iterative conversation step
                if len(st.session_state.text_summary) ==0:
                    first_response_generation(prompt)
                else:
                    iterative_response_generation(prompt)
            else:
                #Checks if the user has given enought content to finalize the story
                if len(st.session_state.text_summary) == 0:
                    #The user has to give more information
                    with st.chat_message("assistant"):
                        st.write("Give me first ideas to generate a story")
                        st.session_state.messages.append({"role": "assistant", "content": "Give me first ideas to generate a story"})
                else:
                    create_final_story()
    except IndexError as e:
        st.write("Start your Story")




if __name__ == "__main__":
    main()
    #Explain the scenery of a fairytale castle on a dark and stormy night.
    #A knight is riding slowly towards the castle

    #explain the scenary of a little boy watching the world cup final in Brazil
    #A few seconds later the TV turned on and in the images of the tv Brazil scored the goal to win the world cup.
