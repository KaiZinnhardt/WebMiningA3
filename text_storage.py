import tiktoken
from openai.types import CompletionUsage

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from nltk.tokenize import sent_tokenize


def summarize_text(prev_text, text, model):
    text_to_summarize = prev_text + "\n\n" + text
    num_tokens = token_counts(text_to_summarize,model)
    num_sentences_per_cluster=10
    #print(text_to_summarize)
    while num_tokens > 3000:
        text_to_summarize = generate_semantic_summary(text_to_summarize, num_sentences_per_cluster=num_sentences_per_cluster)
        num_sentences_per_cluster -=1
        num_tokens = token_counts(text_to_summarize,model)
    return text_to_summarize



def token_counts(text,model):
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

def summarize_text_for_image(text_summary, response_text, num_clusters=4, num_sentences_per_cluster=1):
    text_to_summarize = text_summary + "\n" + response_text
    text_to_summarize = generate_semantic_summary(text_to_summarize,num_clusters=num_clusters, num_sentences_per_cluster=num_sentences_per_cluster)
    return text_to_summarize


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
    storage.summarize_text(text)
    print(storage.get_summary())
    sum_text=storage.summarize_text_for_image(text2,4,1)
    print("\n\n Adding text2 \n\n")
    print(sum_text)