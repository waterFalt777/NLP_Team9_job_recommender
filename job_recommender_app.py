import streamlit as st
import process_data as pda
import pandas as pd
import pca_chart as pc
import matplotlib.pyplot as plt
import word_similarity
import pickle
import re

#Introduce App
st.title('Zot Jobs for MSBA Students 💼🐜🍽️')
st.markdown('(Non-Technical Business Roles in 60 - 120k Salary Range + Data Scientists)')
st.sidebar.markdown("See which jobs best match your profile and optimize your resume / LinkedIn!")
st.sidebar.markdown("This app has 3 functionalities:")
st.sidebar.markdown("1. Predict which job type you match most with based on your resume / LinkedIn.")

st.sidebar.markdown("2. Show which job cluster your resume fits within.")

st.sidebar.markdown("3. Help you find which keywords you're missing and matching for your dream job!")

st.sidebar.markdown("Scroll Down to See All Functionalities!")

#Get and transform user's resume or linkedin
user_input = st.text_area("copy and paste your resume or linkedin here", '')

user_input = str(user_input)
user_input = re.sub('[^a-zA-Z0-9\.]', ' ', user_input)  
#STEPH CHANGES: Renamed this user input var since we need the string version to use for calculate_job_similarities()
first_user_input = user_input.lower()

user_input = pd.Series(user_input)

#load NLP + classification models

topic_model = pickle.load(open('topic_model.sav', 'rb'))
classifier = pickle.load(open('classification_model.sav', 'rb'))
vec = pickle.load(open('job_vec.sav', 'rb'))

classes, prob = pda.main(user_input, topic_model, classifier, vec)

data = pd.DataFrame(zip(classes.T, prob.T), columns = ['jobs', 'probability'])

#Plot probability of person belonging to a job class
def plot_user_probability():
    #plt.figure(figsize = (2.5,2.5))

    #ORIGINAL
    #plt.barh(data['jobs'], data['probability'], color = 'r')
    #plt.title('Percent Match of Job Type')
    
    #STEPH 
    zippedDatalst = list(zip(data['jobs'],data['probability']))
    # Sorting by second element
    highest_prob_lst = sorted(zippedDatalst, key=lambda x: -x[1],)
    #print(highest_prob_lst)
    
    #plotting bar graph

    data2 = data.sort_values(by=data.columns[1])#, ascending=False)
    plt.barh(data2['jobs'], data2['probability'], color = 'r')
    plt.title('Percent Match of Job Type')
    st.pyplot()
    
    return highest_prob_lst


#Plot where user fits in with other job clusters
def plot_clusters():
    st.markdown('This chart uses PCA to show you where you fit among the different job archetypes.')
    X_train, pca_train, y_train, y_vals, pca_model = pc.create_clusters()
    for i, val in enumerate(y_train.unique()):
        y_train = y_train.apply(lambda x: i if x == val else x)
    example = user_input
    doc = pc.transform_user_resume(pca_model, example)

    pc.plot_PCA_2D(pca_train, y_train, y_vals, doc)
    st.pyplot()



problst = plot_user_probability()


#STEPH
st.title("🔥💼TOP 5 MATCHING JOBS 💼🔥")
#STEPH CHANGES:
'''
   1. Made If condition to check if user input is empty, if so, display message to enter resume
   2. Called the calculate_job_similarities() function to get the top 5 matching jobs
   3. Changed the calculate_job_similarities() to take in the string version of the user input
'''
if first_user_input == '':
    st.write('Please enter your resume above to see your top 5 matching jobs!')
else:
    top_profession = problst[0][0]
    joblst = pda.returnTop5Jobs(top_profession)#"data,analyst")
    job_similarity_lst = pda.calculate_job_similarities(first_user_input, joblst) #changes to userinput as string, formerly as series
    st.write(job_similarity_lst)

#for SCATTER PLOT
st.title('Representation Among Job Types')
plot_clusters()

st.title('Find Matching Keywords')
st.markdown('This function shows you which keywords your resume either contains or doesnt contain, according to the most significant words in each job description.')
st.markdown("The displayed keywords are stemmed, ie 'analysis' --> 'analys' and 'commision' --> 'commiss'")
option = st.selectbox(
    'Which job would you like to compare to?',
 ('ux,designer', 'data,analyst', 'project,manager', 'product,manager', 'account,manager', 'consultant', 'marketing', 'sales',
 'data,scientist'))

st.write('You selected:', option)
matches, misses = word_similarity.resume_reader(user_input, option)
match_string = ' '.join(matches)
misses_string = ' '.join(misses)

st.markdown('Matching Words:')
st.markdown(match_string)
st.markdown('Missing Words:')
st.markdown(misses_string)


#STEPH - REMOVE ERROR
st.set_option('deprecation.showPyplotGlobalUse', False)