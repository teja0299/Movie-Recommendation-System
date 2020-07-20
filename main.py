#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import cloudpickle
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process


# In[ ]:


app = Flask(__name__)

#loading "model.pkl" to get our latent dataframes

with open("model.pkl","rb") as f:
    latent_matrix_1_df = cloudpickle.load(f)
    latent_matrix_2_df = cloudpickle.load(f) 


# In[ ]:


movies = pd.read_csv("movies.csv")


# In[ ]:
def fw(m):
    temp = process.extractOne(m,movies['title'])[0]
    a_1 = np.array(latent_matrix_1_df.loc[temp]).reshape(1,-1)
    a_2 = np.array(latent_matrix_2_df.loc[temp]).reshape(1,-1)

        # calculate the similarities of this movie with others in the list
    score_1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
    score_2 = cosine_similarity(latent_matrix_2_df, a_2).reshape(-1)

        # an average measure of both content and collaborative gives hybrid
    hybrid = (score_1 + score_2)/2.0
    dictDf  = {'hybrid': hybrid }
    similar = pd.DataFrame(dictDf, index= latent_matrix_1_df.index)
        
        #sort it on basis of either of the three methods

    similar.sort_values('hybrid',ascending=False, inplace=True)
    similar_movies_list = list(similar.head(30).index)

    return similar_movies_list
    
    

def rcmd(m):
    
    if m not in movies['title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
        
    else:
        a_1 = np.array(latent_matrix_1_df.loc[m]).reshape(1,-1)
        a_2 = np.array(latent_matrix_2_df.loc[m]).reshape(1,-1)

        # calculate the similarities of this movie with others in the list
        score_1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
        score_2 = cosine_similarity(latent_matrix_2_df, a_2).reshape(-1)

        # an average measure of both content and collaborative gives hybrid
        hybrid = (score_1 + score_2)/2.0
        dictDf  = {'hybrid': hybrid }
        similar = pd.DataFrame(dictDf, index= latent_matrix_1_df.index)
        
        #sort it on basis of either of the three methods

        similar.sort_values('hybrid',ascending=False, inplace=True)
        similar_movies_list = list(similar.head(30).index)

        return similar_movies_list


# In[ ]:


#home page of the app
@app.route("/")
def home():
    return render_template('home.html')


# In[ ]:


@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')


# In[ ]:

@app.route("/again_recommend")
def again_recommend():
    moviee = request.args.get('moviee')
    a = fw(moviee)
    return render_template('again_recommend.html',moviee=moviee,a=a,t='l')
    



# In[ ]:
if __name__ == '__main__':
    app.run(debug=True)



