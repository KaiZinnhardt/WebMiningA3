# WebMiningA3
***Create Fairytale Stories while using the OpenAI API***
## Setup Guide:
1. Place the files ``main.py``, ``text_helper.py``, and ``requirements.txt`` in your IDE. Alternatively, download the following Github repository: https://github.com/KaiZinnhardt/WebMiningA3.git
2. Download all the necessary libraries, which are placed in the ``requirements.txt`` file. The installation of these files can be done by executing the following command: ``pip install -r requirements.txt``
3. Create a folder called streamlit in your working directory and create the  ``secrets.toml`` file in the streamlit folder.
4. Place your OpenAI key in this file. Note: please name the key variable ``OPENAI_API_KEY``.
5. Run the streamlit application by running the ``streamlit run main.py``

## Prompts:
In the first prompt, please set a scene where the story should take place. The response will contain questions on how to continue the story. When using these questions to continue the story, please elaborate thoroughly on the question so that the bot clearly knows what to answer. Nevertheless, it is also possible to extend the story without the usage of the generated questions.

