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
import altair as alt #donut chart






# UI Stuff
#Wide layout
st.set_page_config(layout="wide")

# Load custom CSS from styles.css
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


#universal var
top5matchedJobs = []

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
    user_input = st.sidebar.text_area("Or copy and paste your resume", '')

user_input = str(user_input)
user_input = re.sub('[^a-zA-Z0-9\.]', ' ', user_input)
str_user_input = user_input.lower()

user_input = pd.Series(str_user_input)



#Pre-Coded by the Owner: Load NLP + classification models
 
#1. Breaks down the resume into related job topics 
topic_model = pickle.load(open('topic_model.sav', 'rb'))

#2. Takes the related job topics and classify resume into the best job class 
classifier = pickle.load(open('classification_model.sav', 'rb')) 

#3.Vectorize the resume after tokenization)
vec = pickle.load(open('job_vec.sav', 'rb')) 

#4. Get the job classes and their probabilities from another function <main()> in another file <process_data.py> 
# and stores the output in classes and prob
classes, prob = pda.main(user_input, topic_model, classifier, vec) 

#5. Create a dataframe with the job classes and their probabilities
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



#CONTAINERS IN DASHBOARD

c1, c2, c3 = st.columns((3,4,3))
with c1:
    
  with st.container():
    #st.header('Top Matching Job Types')
    st.header3 = st.markdown(f"<h2 style='text-align: center; color: white;'>Top Matching Job Types</h2>", unsafe_allow_html=True)
    if str_user_input == "":
        st.write("Please enter your resume in the text box above or upload a PDF file")
    else:
        problst = plot_user_probability()
    
        
    with st.container():
        #st.header2('Representation Among Job Types')
        st.header3 = st.markdown(f"<h2 style='text-align: center; color: white;'>Representation Among Job Types</h2>", unsafe_allow_html=True)
        if str_user_input != "":
            plot_clusters()



with c2:
    #STEPH
    if str_user_input == "":
        st.markdown(
            f"<h1 style='text-align: center; color: white;'>TOP MATCHING JOBS For <span style='color: yellow;'> ... </span>:</h1>",
            unsafe_allow_html=True)
        st.write("Please enter your resume in the text box above or upload a PDF file")
    else:
        top_profession = problst[0][0]
        readable_profession_name = convert_job_names(top_profession)
        st.markdown(
            f"<h1 style='text-align: center; color: white;'>TOP MATCHING JOBS For <span style='color: yellow;'>{readable_profession_name}</span>:</h1>",
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
            st.markdown(
                        f"""
                            <div class="job-card">
                                <div class="job-title">{job['Job Title']}</div>
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
                

with c3:
    #st.header2("Job Match Percentage")
    st.header3 = st.markdown(f"<h2 style='text-align: center; color: white;'>Job Match Percentage</h2>",unsafe_allow_html=True)
     # Function to create a donut chart
    def create_donut_chart(match_job, match_percentage):
            remaining_percentage = 100 - match_percentage
            source = pd.DataFrame({
                'Category': ['Match', 'Remaining'],
                'Value': [match_percentage, remaining_percentage]
            })
            st.header3 = st.markdown(f"<h4 style='text-align: center; color: white;'>{match_job}</h4>", unsafe_allow_html=True)
            chart = alt.Chart(source).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Value", type="quantitative"),
                color=alt.Color(field="Category", type="nominal",
                                scale=alt.Scale(domain=['Match'],
                                                range=['#32de84'])),#, '#E0E0E0'])), #4CAF50- original green color
                tooltip=['Category', 'Value']
            ).properties(
                width=150,
                height=150
            )

             # Add percentage text in the middle
            text = alt.Chart(pd.DataFrame({'text': [f"{match_percentage}%"]})).mark_text(
                size=20, color='white', fontWeight='bold'
            ).encode(
                text='text:N'
            )

            return chart + text 
     # Generate and display the donut chart
    if str_user_input != "":
        for i, (index, job) in enumerate(top5matchedJobs.iterrows()):
            jobTitle = job[3] #takes the job title
            jobMatch = job[4] #takes the normalized similarity score
            donut_chart = create_donut_chart(jobTitle, jobMatch) #passing in each job match %
            st.altair_chart(donut_chart, use_container_width=True)





#STEPH - REMOVE ERROR
st.set_option('deprecation.showPyplotGlobalUse', False)