import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
from numpy.random import seed
from numpy.random import rand
from sklearn.decomposition import PCA
from tensorflow import keras
from PIL import Image
pca = PCA(n_components=1)
# seed random number generator
seed(1)
import zipfile
import os


# Load the pickled model

model_cd = keras.models.load_model('pickeled models\model_cd.h5')

model_cu = keras.models.load_model('pickeled models\model_cu.h5')

model_hg = keras.models.load_model('pickeled models\model_hg.h5')

model_pb = keras.models.load_model('pickeled models\model_pb.h5')

def preprocessing(data):
    
  x=np.array(data[:,0])
  y=np.array(data[:,1])
  xnew=np.mean(x)+rand(5600-len(x))*np.std(x)
  xnew=np.sort(xnew)
  f = interpolate.interp1d(x, y)
  ynew=f(xnew)
  xnew=xnew.reshape(len(xnew),1)
  ynew=ynew.reshape(len(ynew),1)

  con=np.concatenate((xnew,ynew),axis=1)
  data=np.concatenate((data,con),axis=0)
  data= pd.DataFrame(data)
   
  data=data.sample(n=200,replace=True)
  data=np.array(data)
  ind = np.argsort( data[:,0] )
  data = data[ind]
  
   

  return data

        
    # Define the Streamlit app
def main():
    # Set the title of the app
    st.title('Heavy Metal Detection')
    

    img = Image.open("FLOW DIAGRAM.png")
    st.image(img, caption="Scheme For modelling used", use_column_width=True)
    # Add a file uploader widget
    uploaded_files = st.file_uploader("Upload XLSX files", type=["xlsx"], accept_multiple_files=True)
    # If a file is uploaded
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:


        
        
            
            # Read the file into a Pandas DataFrame
            df = pd.read_excel(uploaded_file)
            # Plot the data using Matplotlib
            df=df.iloc[1600:,:]
            fig, ax = plt.subplots()
            ax.plot(df.iloc[:,0], df.iloc[:,1])
            ax.set_xlabel('voltage')
            ax.set_ylabel('current')
            ax.set_title(uploaded_file.name)
            st.pyplot(fig)
            data=preprocessing(np.array(df))
            pca.fit(data)
            data=pca.transform(data)
            data=data.reshape(200,1)
            # Make predictions using the pickled model
            if model_cd.predict(data.reshape(1,200,1))[0]>0.6:
                st.write("Cadmium is Present")
            if model_cu.predict(data.reshape(1,200,1))[0]>0.6:
                st.write("Copper is Present")
            if model_hg.predict(data.reshape(1,200,1))[0]>0.6:
                st.write("Mercury is Present")
            if model_pb.predict(data.reshape(1,200,1))[0]>0.6:
                st.write("Lead is Present")
            
        
if __name__ == '__main__':
    main()
