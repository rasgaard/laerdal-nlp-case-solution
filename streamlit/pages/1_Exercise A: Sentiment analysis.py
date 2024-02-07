import streamlit as st
from streamlit import components

st.write("""## Data breakdown

Two data sets are expected to be used in this case. The COCO dataset and the Senticap dataset.
         
1. The COCO dataset contains images and captions. 
2. The Senticap dataset contains captions and sentiment labels.

My initial thoughts are that the Senticap dataset can be used to train a supervised classification model to predict the sentiment of the captions in the COCO dataset as per the readme in the GitHub repo:

> the coco captions are more used for model prediction in this project. We get the annotations to train and evaluate a sentiment model from the Senticap data if this is needed.
""")

st.write("""## Sentiment Analysis
For this task I will use the [SetFit library](https://github.com/huggingface/setfit) to train a model on the Senticap dataset. The model will then be used to predict the sentiment of the captions in the COCO dataset. I originally went to look for a model that was already fine-tuned on Senticap but was unable to find anything which motivated me to train a model myself.
         
The reason why I didn't just go for an off-the-shelf sentiment analysis model was that I wanted the model to be trained on data that closely mimics the distribution of data seen in the COCO captions. If the model was trained on a totally different data distribution I would expect the model to perform poorly on the COCO captions due to the domain shift.

Using only 32 labels for each of the two classes (positive and negative) the model obtained a **95% accuracy** on the entire Senticap dataset (~9000 samples) and took about 2 minutes to train due to the low amount of training data. The model is made available on the HuggingFace Hub: [`rasgaard/setfit-senticap`](https://huggingface.co/rasgaard/setfit-senticap). 

The `SetFit` package offers an efficient way to fine-tune sentence-transformers by using a contrastive loss. Each model trained using SetFit consists of a model body and model head. 
         
1. The model body is the sentence transformer model, which is a pre-trained model that takes in a sentence and outputs a fixed-size embedding vector.
         
2. The model head is a simple logistic regression model which takes in the embedding vector and outputs a prediction for the sentiment of the sentence.


Intercepting the embedding vectors from the model body we can visualize the captions in 2D space using UMAP to see how the captions are distributed. The blue dots are positive captions while the red dots are negative. It is clear that the model has learned to separate the two classes quite well.
""")

c1, c2, c3 = st.columns(3)
with c2:
    st.components.v1.html(open("../img_data/coco_ann2014/annotations/umap_embeddings.html", 'r').read(), width=600, height=600)
