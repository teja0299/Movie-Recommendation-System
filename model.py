#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle


# In[ ]:


ratings_f1 = pd.read_csv("ratings_f1.csv")
final = pd.read_csv("final.csv")
movies = pd.read_csv('movies.csv')


# # creating a content based latent matrix from movie metadata :
# # tf-idf vectors and truncated SVD :

# In[ ]:


#tf --> term frequency
#idf --> inverse data frequency

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = 'english')
tfidf_matrix = tfidf.fit_transform(final['metadata'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),index = final.index.tolist())

# here tfidf_df is a sparse dataframe(sparse means contains more 0;s than a data ) with very large dimesnsions
# so dimension reduction is done with SVD


# In[ ]:


#compressing with SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=200)
latent_matrix = svd.fit_transform(tfidf_df)


# In[ ]:


# number of latent dimensions to keep
n = 200
latent_matrix_1_df = pd.DataFrame(latent_matrix[:,0:n], index = final.title.tolist())


# # creating a collaborative based latent matrix from user ratings

# In[ ]:


ratings_f2 = ratings_f1.pivot(index= 'movieId', columns='userId',values='rating').fillna(0)


# In[ ]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=200)
latent_matrix_2 = svd.fit_transform(ratings_f2)
latent_matrix_2_df = pd.DataFrame(latent_matrix_2,index = final.title.tolist())


# # dumping the latent matrices dataframe with Pickle

# In[ ]:


with open("model.pkl","wb") as f:
    cloudpickle.dump(latent_matrix_1_df,f)
    cloudpickle.dump(latent_matrix_2_df,f)


# In[ ]:




