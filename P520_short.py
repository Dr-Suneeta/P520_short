import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle as pickle
import seaborn as sns




# Title
st.title("**Product Recommendation Engine**")
st.write("### **Data Preview**")



# Data import with exception handling
try:
    df = pd.read_csv("rating_short.csv")
    st.write(df.head(5))
except FileNotFoundError:
    st.error("Ratings CSV file not found. Please check the path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()


# dtype correction
df["date"] = pd.to_datetime(df.date, unit="s")
df["rating"] = df.rating.astype("int8")


# datewise upsampling
rat=df[["date","rating"]]
rat.set_index(rat.columns[0], inplace=True)
rat = rat.resample("D").mean()
rat=rat.fillna(rat.rating.mean())

# Model Loading with Error Handling
def load_model(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model {file_name} not found. Please check the path.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model {file_name}: {e}")
        st.stop()

ari = load_model('ARI.pkl')
LG= load_model('LG.pkl')




# labelencoding for userid
LE_uid=LabelEncoder()
LE_pid=LabelEncoder()
df["userid"]=LE_uid.fit_transform(df.userid)
df["productid"]=LE_pid.fit_transform(df.productid)






# Rating timeline
st.write("### **Ratings Timeline**")
if st.button("#### **Timeline**"):
    yforecast = ari.forecast(steps=500)
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(rat, color='black')
    ax.plot(yforecast, color='green')   
    st.pyplot(fig)



# Rating forecast
st.write("### **Ratings Forecast**")
s = st.slider("**Select the number of days to be forecasted:**", 2,11,1)
if st.button("**Forecast**"):
    yforecast = ari.forecast(steps=s)
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(yforecast, color='green', marker='*', markersize=20, markeredgecolor="red")
    st.write("### **Rating Forecast**")
    st.pyplot(fig)




# Cluster plotting
st.write("Cluster destribution")

c=st.slider("**Select the number of Clusters:**", 2,11,1)
if st.button("**Show Clusters**"): 
    data=df.drop('date',axis=1).sample(frac=0.01,random_state=42)
    SS=StandardScaler()
    dfs=SS.fit_transform (data)
    pca=PCA(n_components=0.75,random_state=42)
    p=pca.fit_transform(dfs)
    dfp=pd.DataFrame(p)
    KM=KMeans(n_clusters=c,random_state=4)
    KM.fit(dfp)
    fig,ax=plt.subplots(figsize=(10,5))
    sns.scatterplot(x=dfp[0],y=dfp[1],hue=KM.predict(dfp),palette='viridis',ax=ax)
    ax.set_title("Clusters distribution")
    st.pyplot(fig) 
    
   
# Customer Selection
st.write("### **Customized recommendations**")
u = st.selectbox("**Select Customer**",LE_uid.inverse_transform(df.userid.unique()))


# Top N recommendation using LightGBM
st.write("##### **Top n Recommendations**")

n=st.slider("**Select the number of products to be recommended:**", 1,11,1)


# making predicted rating column for the selected user against each unique product id
if st.button("**Top Product Recommended**"):   
    col2 = df.productid.unique()
    col1 = LE_uid.transform([u] * len(col2))
    dfr = pd.DataFrame({'user': col1, 'product': col2})   
    dfr['rating'] = LG.predict(dfr)  
    dfr.sort_values(by='rating', ascending=False, inplace=True)
    dfr = dfr.head(n)
    recommendations = LE_pid.inverse_transform(dfr['product'])
    st.write(f"**Top recommended products for customer {u} are:**", recommendations)



