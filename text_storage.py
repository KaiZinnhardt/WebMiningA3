import tiktoken
from openai.types import CompletionUsage

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from nltk.tokenize import sent_tokenize


def summarize_text(prev_text, text, model,max_tokens=2200):
    """
    Summarizes the generated text with the previously given responses of openai
    :param prev_text: a string containing the previously generated response
    :param text: a string containg the response
    :param model: the model of openai that is being used to create the responses
    :return: generate a string containing the summary
    """
    # concatenate the text_summary with the response_text
    text_to_summarize = prev_text + "\n\n" + text
    #Count the number of tokens that openai recognizes
    num_tokens = token_counts(text_to_summarize,model)
    #Set the number of clusters and sentences for each cluster
    num_sentences_per_cluster=15
    num_clusters=12
    #Loops as long as the number of tokens is bigger than xxx
    while num_tokens > max_tokens:
        # generate the summary
        text_to_summarize = generate_semantic_summary(text_to_summarize, num_clusters=num_clusters, num_sentences_per_cluster=num_sentences_per_cluster)
        #reduce dimensionality of summary
        if num_sentences_per_cluster*2 < num_clusters:
            num_clusters -=1
        else:
            num_sentences_per_cluster -=1
        #counts the number of tokens that openapi recognizes
        num_tokens = token_counts(text_to_summarize,model)

    return text_to_summarize



def token_counts(text,model):
    """
    counts the number of tokens in a text message that openai recognizes
    :param text: The text to count the tokens for
    :param model: the openai model that is used
    :return: the integer number of tokens
    """
    encoding = tiktoken.encoding_for_model(model)
    encodedString = encoding.encode(text)
    return len(encodedString)

def calculateCosts(completionUsage):
    # Price Per Token
    if isinstance(completionUsage, CompletionUsage):
        inputTokenPrice = 0.00001
        outputTokenPrice = 0.00003
        return completionUsage.prompt_tokens * inputTokenPrice + completionUsage.completion_tokens * outputTokenPrice

# Function to generate a semantically ordered summary
def generate_semantic_summary( text, num_clusters=7, num_sentences_per_cluster=10):
    """
    Generates a semantic summary of the text input that is given. This code was created in collaboration with chatgpt.
    :param text: the test to create a summary for
    :param num_clusters: the number of clusters that the K-means algorithm should use to cluster the text
    :param num_sentences_per_cluster: the number of sentences for each cluster the summary should conatin
    :return:
    """
    # Load pre-trained sentence embeddings model
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Encode sentences into embeddings
    sentences = sent_tokenize(text)
    sentence_embeddings = model.encode(sentences)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(sentence_embeddings)

    # Group sentences by clusters
    cluster_sentences = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_sentences:
            cluster_sentences[label] = []
        cluster_sentences[label].append(sentences[i])

    # Select representative sentences from each cluster
    summary_sentences = []
    for cluster_id, sentences in cluster_sentences.items():
        top_sentences = sorted(sentences, key=lambda x: len(x.split()), reverse=True)[:num_sentences_per_cluster]
        summary_sentences.extend(top_sentences)


    # Order summary sentences based on their original positions in the text
    ordered_summary = sorted(summary_sentences, key=lambda x: text.index(x))

    # Join the summary sentences to form the final summary
    summary = '\n'.join(ordered_summary)

    return summary

from transformers import BartForConditionalGeneration, BartTokenizer
def text_summarizer_v2(text):

    print("loading")
    # Load pre-trained BART model and tokenizer
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    print("Done loading")
    # Prepare input for summarization
    #text = "Once upon a time, in a faraway kingdom, there was a brave prince who embarked on a quest to rescue a princess from an evil dragon."

    # Tokenize input text
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)

    # Generate summary using pooled_output from BERT
    summary_ids = model.generate(inputs['input_ids'], num_beams=10, length_penalty=1.5, early_stopping=True)

    # Decode the summary tokens back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_text_for_image(text_summary, response_text, num_clusters=4, num_sentences_per_cluster=1):
    """
    Summarize the text for the image prompt
    :param text_summary: previous text summary
    :param response_text: newly created content
    :param num_clusters: number of clusters that the summary should build
    :param num_sentences_per_cluster: number of sentences for each clusters that the summary should contain
    :return: the final summary as a string
    """
    #concatenate the text_summary with the response_text
    text_to_summarize = text_summary + "\n" + response_text
    #generate the summary
    text_to_summarize = generate_semantic_summary(text_to_summarize,num_clusters=num_clusters, num_sentences_per_cluster=num_sentences_per_cluster)
    #return the summary
    return text_to_summarize

def max_token_calculator(message,model,max_tokens=4000):
    list_message = str(message)
    tokens_used = token_counts(list_message,model)
    setup_loss = 10
    return max_tokens-tokens_used -setup_loss

if __name__ == '__main__':
    #storage = TextStorage("gpt-3.5-turbo")
    text ="""Once upon a time, in a faraway land, there was a magical castle that stood tall and proud against the stormy night sky. The castle was made of shining stones that sparkled in the moonlight, and it had towering turrets that reached up to the stars.

Outside, rain poured down and lightning flashed across the sky, but inside the castle walls, all was warm and cozy. The flickering candlelight cast dancing shadows on the walls, making the castle feel like a mysterious and enchanting place.

As the storm raged on, the castle seemed to come alive with secrets and stories. The wind howled through the corridors, and the old oak doors creaked open and closed on their own. But the brave knights and noble princesses who lived in the castle were not afraid, for they knew that they were safe and protected by the magic that surrounded them.

And so, as the storm rumbled on outside, the inhabitants of the castle snuggled up in their beds, listening to the raindrops tap against the windows and the thunder rumble in the distance. And as they drifted off to sleep, they knew that they were living in a fairytale world where anything was possible, even on the darkest and stormiest of nights. The end. 🏰🌧️⚡
"""
    text2="""As the knight approached the towering castle, he was greeted by a group of magical creatures who had been waiting for his arrival. The creatures, a mischievous fairy, a wise old wizard, and a gentle giant, had been searching for a brave soul to join them on a thrilling adventure to save the kingdom from an evil sorcerer.

The knight, drawn in by the warmth of their friendship and intrigued by the promise of magic, eagerly accepted their invitation. Together, they embarked on a journey filled with perilous challenges and enchanted landscapes, where their bond of friendship was tested and strengthened with each passing day.

Through their combined courage and determination, they finally reached the sorcerer's lair, where a fierce battle of good against evil unfolded. The knight's bravery, the fairy's clever tricks, the wizard's powerful spells, and the giant's strength proved to be a formidable team as they fought against the dark forces threatening the kingdom.

In the end, the power of their friendship and the magic that bound them together was enough to defeat the sorcerer and bring peace back to the land. The knight, forever grateful for the adventure and the companionship of his new friends, knew that their bond would endure long after their heroic quest had come to an end."""
    #summarize_text(text)
    #sum_text=summarize_text_for_image(text2,4,1)
    #print("\n\n Adding text2 \n\n")
    #print(sum_text)
    print("Running summarizer")
    print(text_summarizer_v2(text))
    print("Running summarizer 2")
    print(text_summarizer_v2(text2))
    #list_of_dicts = [{'a': 1, 'b': 2}, {'c': 3, 'd': 4}, {'e': 5, 'f': 6}]
    #list_str = str(list_of_dicts)
    #import json
    #list_json = json.dumps(list_of_dicts)
    #print(type(list_str))
    #print(type(list_json))


