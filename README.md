# text-summarisation

Text summarisation is very useful for daily life purposes whenever you wish not to read the whole text but get a gist of it in less words. There are basically 2 ways of text summarisation which are extractive and abstractive.
Extractive is basically copying and pasting the important lines of the document that will be able to explain the whole document whereas abstractive uses encoder-decoder model to generate a new summary of the given document.

This article quite well explains the text-summarisation "https://towardsdatascience.com/text-summarization-in-python-76c0a41f0dc4?gi=cf398a9544c9"

In this method, I have summarised text using the extractive method by combining the best 2 summarisers available (gensim algo and nltk text-rank) in under 100 words(You can change it if you want to). You can read about them online, I am not including them to only include the required information.

To run the code:

1) Change the dataset location and the destination location for where you want to access the dataset and write the summarised files in the lines 7 & 8.

2) Change the no. of documents of the dataset you want to get the summary of in the line 19. The default is all the articles.

3) In the lines 20,21,22 you can change the the lines that you want to make a summary of rather than making of whole of the document

Also you might have to download the nltk's averaged_perceptron_tagger, so just uncomment the line 6 in the hybrid.py


