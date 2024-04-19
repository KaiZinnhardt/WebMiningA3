import tiktoken
from openai.types import CompletionUsage

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer


class TextHelper:
    def __init__(self,APImodel):
        self.APImodel=APImodel
    def summarize_text(self,prev_text, text,max_tokens=2200):
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
        num_tokens = self.token_counts(text_to_summarize)
        #Set the number of clusters and sentences for each cluster
        num_sentences_per_cluster=8
        num_clusters=20
        #Loops as long as the number of tokens is bigger than xxx
        while num_tokens > max_tokens:
            # generate the summary
            text_to_summarize = self.generate_semantic_summary(text_to_summarize, num_clusters=num_clusters, num_sentences_per_cluster=num_sentences_per_cluster)
            #reduce dimensionality of summary
            if num_sentences_per_cluster*2 < num_clusters:
                num_clusters -=1
            else:
                num_sentences_per_cluster -=1
            #counts the number of tokens that openapi recognizes
            num_tokens = self.token_counts(text_to_summarize)

        return text_to_summarize

    def token_counts(self,text):
        """
        counts the number of tokens in a text message that openai recognizes
        :param text: The text to count the tokens for
        :param model: the openai model that is used
        :return: the integer number of tokens
        """
        encoding = tiktoken.encoding_for_model(self.APImodel)
        encodedString = encoding.encode(text)
        return len(encodedString)

    def calculateCosts(self,completionUsage):
        # Price Per Token
        if isinstance(completionUsage, CompletionUsage):
            inputTokenPrice = 0.00001
            outputTokenPrice = 0.00003
            return completionUsage.prompt_tokens * inputTokenPrice + completionUsage.completion_tokens * outputTokenPrice

    # Function to generate a semantically ordered summary
    def generate_semantic_summary(self, text, num_clusters=7, num_sentences_per_cluster=10):
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

    def summarize_text_for_image(self, text_summary, response_text, num_clusters=4, num_sentences_per_cluster=1):
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
        text_to_summarize = self.generate_semantic_summary(text_to_summarize,num_clusters=num_clusters, num_sentences_per_cluster=num_sentences_per_cluster)
        #return the summary
        return text_to_summarize

    def max_token_calculator(self,message,max_tokens=4000):
        """
        Calculates the maximum number of tokens that the answer can give witout exceeding the context window
        :param message: the input message to the LLM. This is given as a String
        :param model: The model that is used for the OpenAI interaction
        :param max_tokens: Then number of tokens that the context window is allowed to have. Is set to 4000 tokens by defaul
        :return: An integer number of how many tokens are allowed.
        """
        #transform a list into a string
        list_message = str(message)
        #Count the number of tokens
        tokens_used = self.token_counts(list_message)
        #Incorporate an overhead loss.
        setup_loss = 20
        #return the maximum number of tokens that are allowed as a response.
        return max_tokens - tokens_used - setup_loss

if __name__ == '__main__':
    text ="""Once upon a time, in a faraway land, there was a magical castle that stood tall and proud against the stormy night sky. The castle was made of shining stones that sparkled in the moonlight, and it had towering turrets that reached up to the stars.

Outside, rain poured down and lightning flashed across the sky, but inside the castle walls, all was warm and cozy. The flickering candlelight cast dancing shadows on the walls, making the castle feel like a mysterious and enchanting place.

As the storm raged on, the castle seemed to come alive with secrets and stories. The wind howled through the corridors, and the old oak doors creaked open and closed on their own. But the brave knights and noble princesses who lived in the castle were not afraid, for they knew that they were safe and protected by the magic that surrounded them.

And so, as the storm rumbled on outside, the inhabitants of the castle snuggled up in their beds, listening to the raindrops tap against the windows and the thunder rumble in the distance. And as they drifted off to sleep, they knew that they were living in a fairytale world where anything was possible, even on the darkest and stormiest of nights. The end. 🏰🌧️⚡
"""
    text2="""As the knight approached the towering castle, he was greeted by a group of magical creatures who had been waiting for his arrival. The creatures, a mischievous fairy, a wise old wizard, and a gentle giant, had been searching for a brave soul to join them on a thrilling adventure to save the kingdom from an evil sorcerer.

The knight, drawn in by the warmth of their friendship and intrigued by the promise of magic, eagerly accepted their invitation. Together, they embarked on a journey filled with perilous challenges and enchanted landscapes, where their bond of friendship was tested and strengthened with each passing day.

Through their combined courage and determination, they finally reached the sorcerer's lair, where a fierce battle of good against evil unfolded. The knight's bravery, the fairy's clever tricks, the wizard's powerful spells, and the giant's strength proved to be a formidable team as they fought against the dark forces threatening the kingdom.

In the end, the power of their friendship and the magic that bound them together was enough to defeat the sorcerer and bring peace back to the land. The knight, forever grateful for the adventure and the companionship of his new friends, knew that their bond would endure long after their heroic quest had come to an end."""
    text3 ="""Once upon a time, in a magical land far, far away, there stood a magnificent fairytale castle. The castle was made of shimmering white stones, with towers that reached up to touch the stars. On this particular night, dark clouds gathered in the sky, and lightning flashed ominously, casting eerie shadows on the castle walls. But inside, the castle was warm and cozy, with flickering candlelight and the sound of crackling fires.

In one of the castle's highest towers, a brave prince named James was preparing for a great adventure. He had heard tales of a mysterious creature that lived in the enchanted forest beyond the castle walls, and he was determined to discover its secrets. With his faithful companion, a magical talking owl named Hoot, by his side, Prince James set out into the stormy night, his heart filled with excitement and a touch of fear.

As they ventured deeper into the forest, the trees seemed to whisper secrets and the wind howled like a ghostly echo. Suddenly, a strange glow appeared ahead, drawing them closer and closer until they stood face to face with the creature. It was a beautiful unicorn with sparkling eyes and a mane that shimmered like gold.

But before Prince James could speak, a deafening roar echoed through the forest, and the ground shook beneath their feet. What could be causing such a commotion? And would Prince James and Hoot be able to uncover the truth behind the unicorn's mysterious presence? The answer lay ahead, as they journeyed deeper into the heart of the enchanted forest, where even greater dangers and wonders awaited them. But that, my dear friends, is a story for another bedtime...

As Prince James stood in awe before the majestic unicorn, the enchantment of the moment surrounded him like a shimmering veil. The forest seemed to hold its breath, every leaf and blade of grass whispering secrets of old. The unicorn's eyes held a mysterious depth, hinting at a destiny intertwined with the prince's own.

In a soft, melodic whisper that only James could hear, the unicorn uttered words of ancient prophecy and untold magic. The prince's heart raced with the weight of this revelation, realizing that his journey was far from over. With a gentle nudge of its horn, the unicorn beckoned him to follow, leading deeper into the heart of the enchanted forest.

But as they ventured further into the shadows, a distant rumble filled the air, signaling a looming danger on the horizon. What trials awaited Prince James and Hoot in this mystical realm? And what role did destiny play in the unfolding of their fate? Only time would tell as they pressed on, ready to face whatever challenges lay in their path, unaware of the looming cliffhanger that awaited them just beyond the next bend in the forest...

As Prince James and the unicorn raced through the enchanted forest, the whispers of destiny grew louder, echoing through the ancient trees. The prince's heart beat in time with the rhythm of their gallop, a sense of urgency building within him. Shadows danced ominously around them, hinting at the ancient prophecy that had brought them together.

Suddenly, the unicorn skidded to a halt at the edge of a shimmering lake, its waters reflecting the silvery moonlight. In the center of the lake stood a solitary island, shrouded in mist and mystery. The unicorn nudged Prince James to dismount, its eyes filled with a mix of determination and sorrow.

As they stepped onto the island, a figure cloaked in darkness emerged from the mist, revealing a chilling truth that would shake the prince to his core. The figure spoke of a destiny intertwined with the prince's own, of a long-forgotten past that held the key to unlocking the future.

With a final cryptic message, the figure vanished into the shadows, leaving Prince James standing alone on the island, his heart heavy with the weight of revelations yet to come. What secrets lay buried in the depths of the enchanted forest, waiting to be uncovered? And what role did destiny truly play in the prince's journey? As the night closed in around him, a sense of foreboding filled the prince's soul, signaling that the greatest challenges were yet to come...

As Prince James gently cradled the sick baby unicorn in his arms, he felt a surge of determination to help heal the magical creature. With the guidance of Hoot and the mystical forest itself, they embarked on a quest to gather rare herbs and enchanted waters known for their healing properties. Each step of their journey deepened the bond of friendship between the prince and the unicorn, their connection fueled by unwavering trust and shared purpose.

Through trials and triumphs, they witnessed the power of magic intertwining with the natural world, bringing hope and renewal to the ailing unicorn. As the final ingredients were gathered and the healing ritual began, a brilliant light enveloped the clearing, shimmering with ancient enchantment.

But just as the last whispers of magic faded into the night, a shadow fell across the forest, a foreboding presence that hinted at a new challenge looming on the horizon. Prince James and his companions sensed that their journey was far from over, the cliffhanger of destiny awaiting them in the depths of the enchanted realm, where mysteries and dangers intertwined in a dance as old as time itself. Would their newfound bond and the magic they wielded be enough to overcome the looming threat and unravel the ultimate secrets hidden within the heart of the enchanted forest? Only time would reveal the answers to their most profound questions...

As the healing herbs and enchanted waters wove their magic around the baby unicorn, a soft light began to emanate from its wound, slowly closing the gash that threatened its very life. Prince James and Hoot watched in awe as the forest itself seemed to lend its healing energies to the mystical ritual, weaving a tapestry of rejuvenation and renewal.

But just as the last vestiges of the healing magic settled over the wounded unicorn, a deep rumbling echoed through the clearing, shaking the earth beneath their feet. A dark shadow descended upon the forest, blotting out the sunlight and casting a pall of foreboding over the enchanted realm. Sensing danger looming close, the prince's heart swelled with courage as he stood protectively before the now-healed unicorn, ready to face whatever malevolent force dared to challenge their newfound bond.

With a resolute gaze, Prince James braced himself for the unknown adversary that lurked beyond the edge of the clearing, his hand hovering over the hilt of his sword. The air crackled with anticipation, the forest holding its breath in ominous silence as a chilling presence crept ever closer, promising a confrontation that would test the trio's courage and resilience to their very limits. And so, with unshakable determination, they prepared to face the looming threat head-on, the cliffhanger of destiny hanging over them like a shadow shrouded in mystery, waiting to unveil its next treacherous chapter in the enchanted tale of courage and magic.
"""
    #list_of_dicts = [{'a': 1, 'b': 2}, {'c': 3, 'd': 4}, {'e': 5, 'f': 6}]
    #list_str = str(list_of_dicts)
    #import json
    #list_json = json.dumps(list_of_dicts)
    #print(type(list_str))
    #print(type(list_json))
    print(storage.summarize_text(text,text3,1000))

