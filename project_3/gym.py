import pandas as pd
import numpy as np  
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
def load_data():
    df = pd.read_csv('Fitbit_dataset.csv')
    return df
st.set_page_config(page_title="Fitbit Dataset Analysis", layout="wide")

def main():
    st.title("Fitbit Dataset Analysis")
    df = load_data()
    st.header("Dataset Overview")