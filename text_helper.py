import tiktoken
from openai.types import CompletionUsage

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize


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

    def list_of_options(self,options):
        # Extract the AI contnent
        JSON_option = options.choices[0].message.content
        try:
            # transform the string into an dictionary
            JSON_option = eval(JSON_option)
            # extract the three options and place them into strings
            try:
                list_option_1 = list(JSON_option.values())[0]
                if "keyword" in list_option_1[0].lower():
                    option_1 = ' '
                else:
                    option_1 = ', '.join(list_option_1)
            except IndexError as e:
                option_1 = ' '
            try:
                list_option_2 = list(JSON_option.values())[1]
                if "keyword" in list_option_2[0].lower():
                    option_2 = ' '
                else:
                    option_2 = ', '.join(list_option_2)
            except IndexError as e:
                option_2 = ' '
            try:
                list_option_3 = list(JSON_option.values())[2]
                if "keyword" in list_option_3[0].lower():
                    option_3 = ' '
                else:
                    option_3 = ', '.join(list_option_3)
            except IndexError as e:
                option_3 = ' '
        except SyntaxError as e:
            option_1 = ' '
            option_2 = ' '
            option_3 = ' '
        return (option_1, option_2, option_3)

if __name__ == '__main__':
    text = "{'List1': ['Courage', 'Determination', 'Adventure'], 'List2': ['Mystery', 'Intrigue', 'Challenge'], 'List3': ['Friendship', 'Bravery', 'Victory']}"
    text_dict = eval(text)
    print(list(text_dict.values())[3])
