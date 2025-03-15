import streamlit as st
import process_data as pda
import pandas as pd
import pca_chart as pc
import matplotlib.pyplot as plt
import word_similarity
import pickle
import re
import PyPDF2

#added
import matplotlib.colors as mcolors
from wordcloud import WordCloud





# UI Stuff
#Wide layout
st.set_page_config(layout="wide")

# Load custom CSS from styles.css
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#Function

def convert_job_names(job_name):
    '''
    Convert job names to more readable format
    '''
    #print("in convert job fun: ", job_name)
    job_name = job_name.replace(',', ' ')
    job_name = job_name.title()

    #remove extra characters such as '[', ']' if in job_name
    job_name = re.sub(r'[^\w\s]', '', job_name)

   
    return job_name


#Introduce App
st.title("Peter's Job Recommendation System 🐜💼 ")
st.sidebar.header('Submit Your Resume 📄')


st.markdown('This dashboard is designed to help you find the best job for you based on your resume')

# Upload resume file
uploaded_file = st.sidebar.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Read the uploaded PDF file
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    user_input = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        user_input += page.extract_text()
else:
    user_input = st.sidebar.text_area("Or copy and paste your resume or LinkedIn here", '')

user_input = str(user_input)
user_input = re.sub('[^a-zA-Z0-9\.]', ' ', user_input)
str_user_input = user_input.lower()

user_input = pd.Series(str_user_input)



#load NLP + classification models
topic_model = pickle.load(open('topic_model.sav', 'rb'))
classifier = pickle.load(open('classification_model.sav', 'rb'))
vec = pickle.load(open('job_vec.sav', 'rb'))

classes, prob = pda.main(user_input, topic_model, classifier, vec)

data = pd.DataFrame(zip(classes.T, prob.T), columns = ['jobs', 'probability'])



#Plot probability of person belonging to a job class
def plot_user_probability():
  
    #STEPH 
    zippedDatalst = list(zip(data['jobs'],data['probability']))
    # Sorting by second element
    highest_prob_lst = sorted(zippedDatalst, key=lambda x: -x[1],)
    
    #plotting bar graph
    data2 = data.sort_values(by=data.columns[1])#, ascending=False)
    # plt.barh(data2['jobs'], data2['probability'], color = 'r')
    converted_job_data = data2
    #change job names to readable format
    converted_job_data['jobs']=converted_job_data['jobs'].apply(convert_job_names)

    #Gradient color map
    cmap = plt.get_cmap('cividis')
    norm = mcolors.Normalize(vmin=converted_job_data['probability'].min(), vmax=converted_job_data['probability'].max())
    colors = cmap(norm(converted_job_data['probability']))

    #Plot
    fig, ax = plt.subplots()
    bars = ax.barh(converted_job_data['jobs'], converted_job_data['probability'], color=colors)

    # Set the background to be transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

     # Customize the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    plt.title('Percent Match of Job Type', color='white')
    # Display the plot in Streamlit
    st.pyplot(fig)
    
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

# Function to generate and display word cloud // BACK BURNER
# def display_wordcloud(text):
#     wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate(text)
#     fig, ax = plt.subplots()
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis('off')
#     st.pyplot(fig)


#CONTAINERS IN DASHBOARD

c1, c2 = st.columns((4,3))
with c1:
    
  with st.container():
    st.header('Top Matching Job Types')
    if str_user_input == "":
        st.write("Please enter your resume in the text box above or upload a PDF file")
    else:
        problst = plot_user_probability()
    
        
    with st.container():
        st.header('Representation Among Job Types')
        if str_user_input != "":
            plot_clusters()



with c2:
    #STEPH
    if str_user_input == "":
        st.markdown(
            f"<h3 style='text-align: center; color: white;'>TOP MATCHING JOBS For <span style='color: yellow;'> ... </span>:</h3>",
            unsafe_allow_html=True)
        st.write("Please enter your resume in the text box above or upload a PDF file")
    else:
        top_profession = problst[0][0]
        readable_profession_name = convert_job_names(top_profession)
        st.markdown(
            f"<h3 style='text-align: center; color: white;'>TOP MATCHING JOBS For <span style='color: yellow;'>{readable_profession_name}</span>:</h3>",
            unsafe_allow_html=True)
        joblst = pda.returnJobsByKeywd(top_profession)
        top5matchedJobs = pda.calculate_job_similarities(str_user_input, joblst)
        #Change names to readable format
        top5matchedJobs['Job'] = top5matchedJobs['Job'].apply(convert_job_names)
        top5matchedJobs['Job Title'] = top5matchedJobs['Job Title'].apply(convert_job_names)
        

      # Display job cards
        for i, (index, job) in enumerate(top5matchedJobs.iterrows()):
            job_description = job[0]
            short_description = ' '.join(job_description.split()[:300]) + '...'
            job_similarity = job[4]
            st.markdown(
                        f"""
                            <div class="job-card">
                                <div class="job-title">{job['Job Title']}</div>
                                <div class="job-similarity" style="color: yellow;">Similarity: {job_similarity}%</div>
                                <div class="job-description">{short_description}</div>
                            </div>
                        """,
                        unsafe_allow_html=True
                )
            with st.expander("Read more"):
                st.markdown(
                        f"""
                            <div class="expanded-job-description">
                                {job[0]}
                            </div>
                        """,
                        unsafe_allow_html=True
                    )


# c1, c2 = st.columns((4,3))
# with c1:
#     #for SCATTER PLOT
#     st.title('Representation Among Job Types')
#     plot_clusters()

#STEPH: words below don't make sense. back burner until i figure a better way of displaying this
# with c2:
#     st.title('Find Matching Keywords')
#     st.markdown('This function shows you which keywords your resume either contains or doesnt contain, according to the most significant words in each job description.')
#     st.markdown("The displayed keywords are stemmed, ie 'analysis' --> 'analys' and 'commision' --> 'commiss'")
#     option = st.selectbox(
#         'Which job would you like to compare to?',
#     ('ux,designer', 'data,analyst', 'project,manager', 'product,manager', 'account,manager', 'consultant', 'marketing', 'sales',
#     'data,scientist'))

#     st.write('You selected:', option)

#     if user_input[0] != "":
#         matches, misses = word_similarity.resume_reader(user_input, option)
#         match_string = ' '.join(matches)
#         misses_string = ' '.join(misses)

        # st.markdown('Matching Words:')
        # display_wordcloud(match_string)  # Display word cloud for matching words
        # st.markdown('Missing Words:')
        # display_wordcloud(misses_string)  # Display word cloud for missing words



#STEPH - REMOVE ERROR
st.set_option('deprecation.showPyplotGlobalUse', False)