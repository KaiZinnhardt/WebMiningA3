#Import to get the helper functions place in the text_storage file
import text_helper

#Import Streamlit and the openai API
import streamlit as st
from openai import OpenAI, BadRequestError

#libraries for displaying images
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO



#Generate unique keys for the 4 display objects
button1_key = "button1"
button2_key = "button2"
button3_key = "button3"
button4_key = "button4"
input1_key = "input1"
input2_key = "input2"


# Set OpenAI API key from Streamlit secrets file in the path ".streamlit/secrets.toml"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
gpt_model = "gpt-3.5-turbo" #"gpt-4"

Helper = text_helper.TextHelper(gpt_model)

def moderation_check(input_message):
    """
    Checks wheather the input has any harmful information that should not be processed by the LLM
    :param input_message: Is the communication stream that the user give the API to generate a response tailored to this content.
    :return: True if the input contains sensored information and false if the input does not contain sensored information
    """
    #transforms the array into a string
    input_message = str(input_message)
    #splits the string after every 2000 characters
    list_message1 = [input_message[i:i+1900] for i in range(0, len(input_message), 1900)]
    #checks iteratively if the message contains any vulnerabilities
    for message in list_message1:
        response = client.moderations.create(input=message)
        if response.results[0].flagged == True:
            return True
    #takes a different split parameter
    list_message2 = [input_message[i:i + 1200] for i in range(0, len(input_message), 1200)]
    # checks iteratively if the message contains any vulnerabilities
    for message in list_message1:
        response = client.moderations.create(input=message)
        if response.results[0].flagged == True:
            return True
    return False



def first_response_generation(prompt):
    """
    This the response to the first prompt that the user has entered
    :param prompt: the user input for which the chatbot will generate text
    """
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        #API call to generate the response
        message = [
                {"role": "system",
                 "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
                {"role": "user", "content": "Describe the context for a bedtime story using the following prompt:"+prompt+"\n Add a cliffhanger at the end of the story."}
            ]
        #calculate the allowed tokens
        allowed_tokens = Helper.max_token_calculator(message)
        #checks if the message is senitized
        mod_bool = moderation_check(message)
        #if it is senitized it should not give an answer
        if mod_bool == True:
            st.write("The input is senitized please use an appropriate prompt")
            return None
        #Perform the API call
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=message,
            stream=True,
            max_tokens=allowed_tokens
        )
        #create the response text as a stream
        response_text = st.write_stream(stream)
        #create an image prompt with summarized data
        image_prompt = "Create an image in a cartoon like style for the following context:" + Helper.summarize_text_for_image('', response_text,5,2)
        # checks if the message is senitized
        mod_bool=moderation_check(image_prompt)
        # if it is senitized it should not give an answer
        if mod_bool == True:
            # Append the text messages into the streamlit session state variable
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.write("The input is for the image senitized please use an appropriate prompt")
            return None
        #Create a spinner to create a visual while waiting for the response
        with st.spinner("Generating image"):
            try:
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
                #display the image
                image = Image.open(BytesIO(requests.get(image_url).content))
                st.image(image)
            except (UnidentifiedImageError, UnboundLocalError) as e: #Catch potential error messages that could occur.
                st.markdown("***Image Couldn't be loaded***")
                image_url = ''
            except BadRequestError as e: #Further Exception that should be caught in the case the moderation doesn't catch a senorized input
                st.markdown("Could not display image. Received the following error message: " + str(e))
                image_url = ''
    #Create a summary of the text, where 'st.session_state.text_summary' is ''
    text_summary = Helper.summarize_text(st.session_state.text_summary, response_text)
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
                {"role": "user", "content": "While taking the summarized fairytale into account; create and describe a scenary in less than 150 words for the following prompt: "+prompt}
            ]
        # calculate the allowed tokens
        allowed_tokens = Helper.max_token_calculator(message)
        # checks if the message is senitized
        mod_bool = moderation_check(message)
        # if it is senitized it should not give an answer
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
        image_prompt = "Create an image in a cartoon like style for the following context:" + Helper.summarize_text_for_image('', response_text, 5, 2)
        # checks if the message is senitized
        mod_bool = moderation_check(image_prompt)
        # if it is senitized it should not give an answer
        if mod_bool == True:
            # Append the text messages into the streamlit session state variable
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.write("The input is for the image senitized please use an appropriate prompt")
            return None
        # Create an interactive graphic while waiting for the image to be generated
        with st.spinner("Generating image"):
            try:
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
                # display the image
                image = Image.open(BytesIO(requests.get(image_url).content))
                st.image(image)
            except (UnidentifiedImageError, UnboundLocalError) as e: #Catch potential error messages that could occur.
                st.markdown("***Image Couldn't be loaded***")
                image_url = ''
            except BadRequestError as e: #Further Exception that should be caught in the case the moderation doesn't catch a senorized input
                st.markdown("Could not display image. Received the following error message: " + str(e))
                image_url = ''
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
        option_test = [
            {"role": "system",
             "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
            {"role": "user",
             "content": "Can you please summarize the previous bedtime story. Later use this summary to extend the Bedtime Story."},
            {"role": "assistant", "content": st.session_state.text_summary},
            {"role": "user",
             "content": "Create and just describe ascenary in less than 150 words for the following prompt, while taking the summary into account: " + prompt},
            {"role": "assistant", "content": response_text},
            {"role": "user",
             "content": "Create three lists, each containing a list of three key words just seperated by commas in JSON format to extend the story, with the following input: " + prompt}
        ]
        allowed_tokens = Helper.max_token_calculator(option_test)
        # checks if the message is senitized
        mod_bool = moderation_check(option_message)
        # if it is senitized it should not give an answer
        if mod_bool == True:
            st.write("The input is senitized please use an appropriate prompt")
            return None
        #API call for generating the first option
        options = client.chat.completions.create(
            model=st.session_state["openai_model"],
            #gives as much contentext as possible while staying cost efficient
            messages= option_test,
            #limits the output space such that only 1-3 keywords are returned
            max_tokens=allowed_tokens
        )
        JSON_option = options.choices[0].message.content
        print(type(JSON_option))
        JSON_option = eval(JSON_option)
        print(JSON_option)
        option_1 = ', '.join(list(JSON_option.values())[0])
        option_2 = ', '.join(list(JSON_option.values())[1])
        option_3 = ', '.join(list(JSON_option.values())[2])
        # Create a summary of the text
        text_summary = Helper.summarize_text(st.session_state.text_summary, response_text)
        # Save the summary in a streamlit session state variable
        st.session_state.text_summary = text_summary
        # Saves the first part of the interaction
        st.session_state.messages.append({"role": "assistant", "content": response_text, "image": image_url, "option1":option_1, "option2":option_2, "option3":option_3})
        #save the prompt
        st.session_state.option_prompt = prompt
        #save the response from the LLM
        st.session_state.iterative_scenary_response =response_text
        #To enable the chat input such that it is necessary to select one of the three choices
        st.session_state.toggle = False

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
                 "content": "Create and just describe a scenary in less than 150 words for the following prompt, while taking the summary into account: " + prompt},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content":"Create three key words just seperated by commas to extend the story, with the following input: " + prompt},
                {"role": "assistant", "content": keywords},
                {"role": "user", "content": "Creatively extend the story from the fairytale summary while taking the keyword of "+keywords+" into acount. Thereby reiterate on the last scenary and add a cliff hanger at the end of the story"}
            ]
        # calculate the allowed tokens
        allowed_tokens = Helper.max_token_calculator(message)
        # checks if the message is senitized
        mod_bool = moderation_check(message)
        # if it is senitized it should not give an answer
        if mod_bool == True:
            st.write("The input is senitized please use an appropriate prompt")
            return None
        #The API Call to generate the response block
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
    text_summary = Helper.summarize_text(st.session_state.text_summary,response_text)
    # Save the summary in a streamlit session state variable
    st.session_state.text_summary = text_summary
    #To make sure that the input field is enabled
    st.session_state.toggle = True


def create_final_story():
    """
    Creates the final story for the bedtime story while taking all of the information into account that it was given.
    """
    # The context where the message will be given in.
    message = [
        {"role": "system",
         "content": "You are a helpful assistant that replies in a tone for a 4 year old kid, for which you create a bedtime story."},
        {"role": "user", "content": "Create a creative, engaging, and extensive bedtime story, with a maximum of 1500 words, for a child while using the following summary: " + st.session_state.text_summary}
    ]
    # calculate the allowed tokens
    allowed_tokens = Helper.max_token_calculator(message)
    # checks if the message is senitized
    mod_bool = moderation_check(message)
    # if it is senitized it should not give an answer
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
        header = "#### Final Story"
        st.write(header)
        # create the response text as a stream
        response_text = st.write_stream(stream)
        # creates the prompt of the image. Thereby, we use a summary of the response to create a cost efficient dialogue
        image_prompt = "Create an image in a cartoon like style for the following context:" + Helper.summarize_text_for_image('', response_text, 5, 2)
        # checks if the message is senitized
        mod_bool = moderation_check(image_prompt)
        # if it is senitized it should not give an answer
        if mod_bool == True:
            # Append the text messages into the streamlit session state variable
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.write("The input is for the image senitized please use an appropriate prompt")
            return None
        # Create an interactive graphic while waiting for the image to be generated
        with st.spinner("Generating image"):
            try:
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
                # display the image
                image = Image.open(BytesIO(requests.get(image_url).content))
                st.image(image)
            except (UnidentifiedImageError, UnboundLocalError) as e: #Catch potential error messages that could occur.
                st.markdown("***Image Couldn't be loaded***")
                image_url = ''
            except BadRequestError as e: #Further Exception that should be caught in the case the moderation doesn't catch a senorized input
                st.markdown("Could not display image. Received the following error message: " + str(e))
                image_url = ''
    #Concatenate the strings
    content = header + " \n\n " + response_text
    # Append the text messages into the streamlit session state variable
    st.session_state.messages.append({"role": "assistant", "content": content, "image": image_url})
    #Disbalbes the chat input bar
    st.session_state.complete_story = True

def complete_story():
    """
    Is a filtering step for prompts that contain the 'complet the story' prompt line
    """
    # Checks if the user has given enough content to finalize the story
    if len(st.session_state.text_summary) == 0:
        # The user has to give more information
        with st.chat_message("assistant"):
            st.write("Give me first ideas to generate a story")
            st.session_state.messages.append(
                {"role": "assistant", "content": "Give me first ideas to generate a story"})
    else:
        create_final_story()


def toggle_input():
    """
    Toggles the chat_input to enable it and disable it in the correct times
    :return: var: True/False always gives back the opposite of the previous state
    """
    st.session_state.toggle = not st.session_state.toggle
    return st.session_state.toggle


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

    #Initialize Finished Story
    if "complete_story" not in st.session_state:
        st.session_state.complete_story = False

    # Initialize prompt for filling in the story after the receiving the key words
    if "option_prompt" not in st.session_state:
        st.session_state.option_prompt = ""

    # Initialize the response to the scenary explanation for filling in the story after the receiving the key words
    if "iterative_scenary_response" not in st.session_state:
        st.session_state.iterative_scenary_response = ""

    # Initialize Finished Story
    if "toggle" not in st.session_state:
        st.session_state.toggle = True

    # Display chat messages from history on app rerun
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            #tries to fetch an image in the message state
            if "image" in message:
                try:
                    #displays the image
                    image = Image.open(BytesIO(requests.get(message["image"]).content))
                    st.image(image)
                except (UnidentifiedImageError, UnboundLocalError, requests.exceptions.MissingSchema) as e: #Catch potential error messages that could occur.
                    st.markdown("***Image Couldn't be loaded***")
            #Displays the option for the last keywords completion, to complete the story
            if i == len(st.session_state.messages) - 1:
                if "option1" in message:
                    # Provides the user with the three options as buttons with a text above showing the number of option it displays
                    st.write("Option 1")
                    st.button(message["option1"], key=button1_key, on_click=create_option_content,
                              args=[st.session_state.option_prompt, message["option1"], st.session_state.iterative_scenary_response])
                if "option2" in message:
                    st.write("Option 2")
                    st.button(message["option2"], key=button2_key, on_click=create_option_content,
                              args=[st.session_state.option_prompt, message["option2"], st.session_state.iterative_scenary_response])
                if "option3" in message:
                    st.write("Option 3")
                    st.button(message["option3"], key=button3_key, on_click=create_option_content,
                              args=[st.session_state.option_prompt, message["option3"], st.session_state.iterative_scenary_response])

    #Create the hint for the user how to complete the story
    st.sidebar.markdown("***Hint***: \n\nTo complete the story please enter: \n\n 'Complete the story'")
    # React to user input
    if prompt:=st.chat_input("Feed me with ideas for a Bedtime story... ", key=input1_key, disabled=(st.session_state.complete_story or toggle_input())):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        if 'complete the story' not in prompt.lower():
            #Decides on whether it is the first conversation on an iterative conversation step
            if len(st.session_state.text_summary) ==0:
                first_response_generation(prompt)
            else:
                iterative_response_generation(prompt)
        else:
            complete_story()
        #Rerun the Application to properly show the additional content.
        st.rerun()




if __name__ == "__main__":
    main()
    #Explain the scenery of a fairytale castle on a dark and stormy night.
    #A knight is riding slowly towards the castle

    #explain the scenary of a little boy watching the world cup final in Brazil
    #A few seconds later the TV turned on and in the images of the tv Brazil scored the goal to win the world cup.
