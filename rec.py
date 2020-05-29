import flask
from flask import Flask
from flask import render_template
from flask import request
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

#---------- MODEL ----------------# 
#Importing the cleaned dataset
reviews=pd.read_csv('reviews3stars.csv')

#Vectorize documents
X=reviews['text']
tfidf = TfidfVectorizer(stop_words='english', min_df=3, max_df=0.8, binary=True)
tfidf_result = tfidf.fit_transform(X.values.astype('U'))

#LSA model
nmf_model = NMF(10)
doc_topic = nmf_model.fit_transform(tfidf_result)

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = Flask(__name__)

@app.route("/")
def hello():
    return ('Hello World')
    

@app.route("/recommender", methods=["POST", "GET"])
def recommender(): 
    string=request.args.get('interest')
    print(string)

    text =[string]
    vt = tfidf.transform(text)
    tt = nmf_model.transform(vt)
    recs = pairwise_distances(tt,doc_topic,metric='cosine').argsort()
    top5=recs[0][0:5].tolist()
    
    top5rest=[]
    for x in top5: 
        rest_row=reviews.loc[x]
        top5rest.append(rest_row['name'])
    
    #print(top5rest)

    return flask.render_template('rec.html',
                                 prediction=top5rest)
   # return flask.jsonify(top5rest)

app.run(debug=True)