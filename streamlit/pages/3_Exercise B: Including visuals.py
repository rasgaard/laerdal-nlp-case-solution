import streamlit as st

st.write("""
# Extracting visual information

For this part of the case I will use the [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) model to extract visual information from the images in the COCO dataset. I have not used these kinds of models before but due to the HuggingFace library it was very easy to get started with a pre-trained model. 
         
Processing the entire COCO dataset took considerably longer than any other task in this case. Having not worked with these kinds of models before I was not sure what to expect in terms of performance. The 40,000 images in the COCO dataset took about **1 hour** to process. However, this was only possible because I violated by own limitations and used a GPU cluster at DTU. The estimated time to process the entire dataset on my MacBook Pro was about **4 hours**. 

         
After generating the captions for the image I again used the `sentence-transformers` library using [a model from the HuggingFace Hub](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) to extract a semantic embedding for both the generated caption as well as the caption provided in the COCO dataset. I was interested in comparing the similarity between the two captions to see if the model would generate semantically similar captions to the ones provided in the dataset. We can see in the histogram below that the similarity scores are quite high - as the cosine similarity is bound between -1 and 1 - which indicates that the model is able to generate semantically similar captions to the ones in the dataset.

Additionally, this would also indicate that adding the generated captions and performing sentiment analysis and topic modelling on a combined dataset would not provide much additional value as the generated captions are so close to the ones in the dataset.

""")
st.components.v1.html(open("../img_data/coco_ann2014/annotations/similarity_scores.html").read(), width=800, height=600)

st.write("""
If anything, we have found that the generated captions are in some cases much more generic than their COCO counterpart. For example, the caption "a train on the tracks" has been found to be generated for 654 images. This is obviously not nearly as helpful as the more specific captions in the COCO dataset. This behaviour also shows up as "a man playing tennis", "a plate of food", and many more. 

""")