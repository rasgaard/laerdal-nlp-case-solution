import streamlit as st

st.write("""## Topic Analysis

For the topic analysis I will also utilize the `sentence-transformers` library under the hood. In this case it will be used as part of the [BERTopic](https://maartengr.github.io/BERTopic/index.html) library. I will use the library as plainly as possible to keep it simple and to avoid spending too much time on this part of the case.

For the purposes of this analysis I will try to keep the number of topics fixed to 200 for the ~200.000 captions in the COCO dataset. Initial testing with letting the topic model decide on a number of topics resulted in a very large number of topics which could dilute the usefulness of the results. Limiting the number of topics to 200 should make it a bit easier to interpret but will also put a lot of captions in the "outlier" category. A bit under 40% of the captions were put in the outlier category when using 200 topics.
""")

st.components.v1.html(open("../img_data/coco_ann2014/annotations/topic_model.html").read(), width=1300, height=700)

st.write("""

""")