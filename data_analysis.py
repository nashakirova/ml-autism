import matplotlib.pyplot as plt
from prepare import dataset
import seaborn as sns

fig, axis = plt.subplots(10, 2, squeeze=False) 
plt.style.use('fivethirtyeight')
data = dataset.copy()

data_autism = data.query('autism==1')
data_nonautism = data.query('autism==0')
# Histogram of country distribution
# plt.style.use('fivethirtyeight')
# plt.hist(data['country_of_res'].dropna(), bins = 100, )
# plt.xlabel('Country of residence')
# plt.xticks(rotation=90)
# plt.ylabel('Diagnosed with autism')
# plt.title('App diagnosed with autism per country')
# plt.show()
def absolute_value(val):
    return val
explode = (0.1, 0)
explode_nonautism = (0, 0.1)
labels=['1','0']
# Let's buld histograms on each of 10 questions to analyze what questions are giving the parametrized answers
# We use only rows where autism = 1 (Class/ASD column is showing the per app diagnosis)
axis[0,0].pie([data_autism['A1_Score'].value_counts()[1], data_autism['A1_Score'].value_counts()[0]], explode = explode, labels=labels)
axis[0,1].pie([data_nonautism['A1_Score'].value_counts()[1], data_nonautism['A1_Score'].value_counts()[0]], explode = explode_nonautism, labels=labels)
axis[0,0].set_xlabel('Definitely agree or slightly agree', fontsize=7)
axis[0,1].set_xlabel('Definitely agree or slightly agree', fontsize=7)
axis[1,0].pie([data_autism['A2_Score'].value_counts()[1], data_autism['A2_Score'].value_counts()[0]], explode = explode, labels=labels)
axis[1,0].set_xlabel('Definitely disagree or slightly disagree', fontsize=7)
axis[1,1].pie([data_nonautism['A2_Score'].value_counts()[1], data_nonautism['A2_Score'].value_counts()[0]], explode = explode_nonautism, labels=labels)
axis[1,1].set_xlabel('Definitely disagree or slightly disagree', fontsize=7)
axis[2,0].pie([data_autism['A3_Score'].value_counts()[1], data_autism['A3_Score'].value_counts()[0]], explode = explode, labels=labels)
axis[2,0].set_xlabel('Definitely agree or slightly agree', fontsize=7)
axis[2,1].pie([data_nonautism['A3_Score'].value_counts()[1], data_nonautism['A3_Score'].value_counts()[0]], explode = explode_nonautism, labels=labels)
axis[2,1].set_xlabel('Definitely agree or slightly agree', fontsize=7)
axis[3,0].pie([data_autism['A4_Score'].value_counts()[1], data_autism['A4_Score'].value_counts()[0]], explode = explode, labels=labels)
axis[3,0].set_xlabel('Definitely disagree or slightly disagree', fontsize=7)
axis[3,1].pie([data_nonautism['A4_Score'].value_counts()[1], data_nonautism['A4_Score'].value_counts()[0]], explode = explode_nonautism, labels=labels)
axis[3,1].set_xlabel('Definitely disagree or slightly disagree', fontsize=7)
axis[4,0].pie([data_autism['A5_Score'].value_counts()[1], data_autism['A5_Score'].value_counts()[0]], explode = explode, labels=labels)
axis[4,0].set_xlabel('Definitely disagree or slightly disagree', fontsize=7)
axis[4,1].pie([data_nonautism['A5_Score'].value_counts()[1], data_nonautism['A5_Score'].value_counts()[0]], explode = explode_nonautism, labels=labels)
axis[4,1].set_xlabel('Definitely disagree or slightly disagree', fontsize=7)
axis[5,0].pie([data_autism['A6_Score'].value_counts()[1], data_autism['A6_Score'].value_counts()[0]], explode = explode, labels=labels)
axis[5,0].set_xlabel('Definitely disagree or slightly disagree', fontsize=7)
axis[5,1].pie([data_nonautism['A6_Score'].value_counts()[1], data_nonautism['A6_Score'].value_counts()[0]], explode = explode_nonautism, labels=labels)
axis[5,1].set_xlabel('Definitely disagree or slightly disagree', fontsize=7)
axis[6,0].pie([data_autism['A7_Score'].value_counts()[1], data_autism['A7_Score'].value_counts()[0]], explode = explode, labels=labels)
axis[6,0].set_xlabel('Definitely agree or slightly agree', fontsize=7)
axis[6,1].pie([data_nonautism['A7_Score'].value_counts()[1], data_nonautism['A7_Score'].value_counts()[0]], explode = explode_nonautism, labels=labels)
axis[6,1].set_xlabel('Definitely agree or slightly agree', fontsize=7)
axis[7,0].pie([data_autism['A8_Score'].value_counts()[1], data_autism['A8_Score'].value_counts()[0]], explode = explode, labels=labels)
axis[7,0].set_xlabel('Definitely agree or slightly agree', fontsize=7)
axis[7,1].pie([data_nonautism['A8_Score'].value_counts()[1], data_nonautism['A8_Score'].value_counts()[0]], explode = explode_nonautism, labels=labels)
axis[7,1].set_xlabel('Definitely agree or slightly agree', fontsize=7)
axis[8,0].pie([data_autism['A9_Score'].value_counts()[1], data_autism['A9_Score'].value_counts()[0]], explode = explode, labels=labels)
axis[8,0].set_xlabel('Definitely disagree or slightly disagree', fontsize=7)
axis[8,1].pie([data_nonautism['A9_Score'].value_counts()[1], data_nonautism['A9_Score'].value_counts()[0]], explode = explode_nonautism, labels=labels)
axis[8,1].set_xlabel('Definitely disagree or slightly disagree', fontsize=7)
axis[9,0].pie([data_autism['A10_Score'].value_counts()[1], data_autism['A10_Score'].value_counts()[0]], explode = explode, labels=labels)
axis[9,0].set_xlabel('Definitely agree or slightly agree', fontsize=7)
axis[9,1].pie([data_nonautism['A10_Score'].value_counts()[1], data_nonautism['A10_Score'].value_counts()[0]], explode = explode_nonautism, labels=labels)
axis[9,1].set_xlabel('Definitely agree or slightly agree', fontsize=7)

top = 0.87
step = 0.08

plt.figtext(0.25,top+0.02,"Autism diagnosed", va="center", ha="center", size=11)
plt.figtext(0.75,top+0.02,"Autism not diagnosed", va="center", ha="center", size=11)
plt.figtext(0.25, top, 'I often notice small sounds when others do not', size=10)
top-=step
plt.figtext(0.25, top, 'I usually concentrate more on the whole picture, rather than the small details', size = 10)
top-=step
plt.figtext(0.25, top, 'I find it easy to do more than one thing at once', size = 10)
top-=step
plt.figtext(0.25, top, 'If there is an interruption, I can switch back to what I was doing very quickly', size = 10)
top-=step
plt.figtext(0.25, top, 'I find it easy to ‘read between the lines’ whensomeone is talking to me', size = 10)
top-=step
plt.figtext(0.25, top, 'I know how to tell if someone listening to me is getting bored', size = 10)
top-=step
plt.figtext(0.25, top, 'When I’m reading a story I find it difficult towork out the characters’ intentions', size = 10)
top-=step
plt.figtext(0.25, top, 'I like to collect information about categories of things', size = 10)
top-=step
plt.figtext(0.25, top, 'I find it easy to work out what someone is thinking or feeling just by looking', size = 10)
top-=step
plt.figtext(0.25, top, 'I find it difficult to work out people’s intentions', size = 10)

fig.tight_layout(h_pad=20.0)

plt.show()

'''
As per the outcome, it seems like questions 1 and 8 give pretty incolclusive results with almost 
same percentages responding towards the autism scoring answer. In the training, we will therefore compare
models that include all 10 features and that only inlcude 2-7, 9-10 score answers.
'''

'''
Let's observe correlation between the features
'''


# df=data.copy()
# df.describe()
# freq = df['country_of_res'].value_counts(normalize=True)
# freq_ethnicity = df['ethnicity'].value_counts(normalize=True)
# freq_relation = df['ethnicity'].value_counts(normalize=True)
# df_v1 = df.copy()
# df_v1.head()
# object_cols = [col for col in df_v1.columns if df_v1[col].dtype == "object"]
# # Map the values to their frequencies
# df_v1['contry-freq'] = df_v1['country_of_res'].map(freq)
# df_v1['ethnicity-freq']=df_v1['ethnicity'].map(freq_ethnicity)
# df_v1['relation-freq']=df_v1['relation'].map(freq_relation)
# df_v2=df.drop("country_of_res", axis=1)
# df_v2=df_v2.drop("ethnicity", axis=1)
# df_v2 = df_v2.drop("age_desc", axis=1)
# df_v2 = df_v2.drop("relation", axis=1)
# df_v3=df_v2.replace({'jundice': "yes"}, 1).replace({'jundice' : "no"}, 0).replace({'austim': "yes"}, 1).replace({'austim' : "no"}, 0).replace({'used_app_before': "yes"}, 1).replace({'used_app_before' : "no"}, 0).replace({'gender':'f'},1).replace({'gender':'m'},0)
# df_v3.head()
# plt.figure(figsize=(10,9))
# ax = sns.heatmap(df_v3.corr(), annot=True, cmap="seismic", annot_kws={"fontsize":12})
# ax.tick_params(axis='both', which='major', labelsize=12)
# plt.tight_layout()
# plt.show()
#plt.savefig('correlation.png')