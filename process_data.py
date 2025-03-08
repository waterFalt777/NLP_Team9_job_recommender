import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle


def tokenize_stem(series):

    tokenizer =TreebankWordTokenizer()
    stemmer = PorterStemmer()
    series = series.apply(lambda x: x.replace("\n", ' '))
    series = series.apply(lambda x: tokenizer.tokenize(x))
    series = series.apply(lambda x: [stemmer.stem(w) for w in x])
    series = series.apply(lambda x: ' '.join(x))
    return series

def display_topics(model, feature_names, no_top_words, topic_names=None):
    '''
    displays topics and returns list of toppics
    '''

    topic_list = []
    for i, topic in enumerate(model.components_):
        if not topic_names or not topic_names[i]:
            print("\nTopic ", i)
        else:
            print("\nTopic: '",topic_names[i],"'")

        print(", ".join([feature_names[k]
                       for k in topic.argsort()[:-no_top_words - 1:-1]]))
        topic_list.append(", ".join([feature_names[k]
                       for k in topic.argsort()[:-no_top_words - 1:-1]]))
    return model.components_, topic_list

def return_topics(series, num_topics, no_top_words, model, vectorizer):
    '''
    returns document_topic matrix and topic modeling model
    '''
    #turn job into series
    series = tokenize_stem(series)
    #transform series into corpus
    ex_label = [e[:30]+"..." for e in series]
    vec = vectorizer(stop_words = 'english')

    doc_word = vec.fit_transform(series)

    #build model
    def_model = model(num_topics)
    def_model = def_model.fit(doc_word)
    doc_topic = def_model.transform(doc_word)
    model_components, topic_list = display_topics(def_model, vec.get_feature_names_out(), no_top_words) #original: vec.get_feature_names()
    return def_model.components_, doc_topic, def_model, vec, topic_list#, topics


def process_data():
    '''
    uses the functions above to read in files, model, and return a topic_document dataframe
    '''
    #read in jobs file and get descriptions
    df = pd.read_csv('jobs.csv')
    #df = df[df.keyword!='marketing']
    jobs_df = pd.DataFrame(zip(df['Job Description'], df['keyword']), columns = ['Description', 'Job'])

    array, doc, topic_model, vec, topic_list  = return_topics(jobs_df['Description'],20, 10, TruncatedSVD, TfidfVectorizer)

    topic_df = pd.DataFrame(doc)
    topic_df.columns = ['Topic ' + str(i+1) for i in range(len(topic_df.columns)) ]

    topic_df['job'] = jobs_df.Job
    #Topic_DF.to_csv('topic_df.csv')
    return topic_df, topic_model, vec, topic_list



#STEPH
def returnJobsByKeywd(keyword):
    '''
    Takes in the user's top keyword and returns the top 5 jobs that belong to the keyword
    '''
    df = pd.read_csv('jobs.csv') #make this universal later
    jobs_df = pd.DataFrame(zip(df['Job Description'], df['Job Title'], df['keyword']), columns=['Description', 'Job Title', 'Job'])
    # Filter rows where the 'Job' column matches the user's keyword
    JobsByKeywd = jobs_df[jobs_df['Job'] == keyword]
   
    
    return JobsByKeywd

#ANI
def calculate_job_similarities(user_input, top_5_jobs_df):
    '''
    Calculate cosine similarity between user input and job descriptions,
    rank jobs, and return similarity scores as percentages
    '''
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Combine user input and job descriptions
    all_text = [user_input] + list(top_5_jobs_df['Description'])
    
    # Create TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(all_text)
    
    # Calculate cosine similarity between user input and each job description
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    # Create a dataframe with jobs and their similarity scores
    similarity_df = pd.DataFrame({
        'Description': top_5_jobs_df['Description'],
        'Job': top_5_jobs_df['Job'],
        'Similarity': cosine_similarities[0] * 100  # Convert to percentage
    })
    
    # Sort by similarity score in descending order
    ranked_jobs = similarity_df.sort_values(by='Similarity', ascending=False)
    
    # Round similarity scores to 2 decimal places
    ranked_jobs['Similarity'] = ranked_jobs['Similarity'].round(2)
    
    return ranked_jobs.head(5)




def predictive_modeling(df):
    '''
    fits, optimizes, and predicts job class based on topic modeling corpus
    '''
    X,y = df.iloc[:,0:-1], df.iloc[:, -1]
    X_tr, X_te, y_tr, y_te = train_test_split(X,y)

    rfc = RandomForestClassifier(n_estimators = 500, max_depth = 9)
    rfc.fit(X_tr, y_tr)
    print('acc: ', np.mean(cross_val_score(rfc, X_tr, y_tr, scoring = 'accuracy', cv=5)))
    print('test_acc: ', accuracy_score(y_te, rfc.predict(X_te)))
    print(rfc.predict(X_te))
    return rfc

def predict_resume(topic_model, model, resume):
    '''
    transforms a resume based on the topic modeling model and return prediction probabilities per each job class
    '''
    doc = topic_model.transform(resume)
    return model.predict_proba(doc), model.classes_

def get_topic_classification_models():
    jobs_df, model, vec , topic_list= process_data()
    model_1 = predictive_modeling(jobs_df)
    return model, model_1, vec


def main(resume, topic_model, predictor, vec):
    '''
    run code that predicts resume
    '''
    
    doc = tokenize_stem(resume)
    doc = vec.transform(doc)
    probabilities, classes = predict_resume(topic_model, predictor, doc)
    return classes, probabilities[0]*100
