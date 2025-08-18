# Show a slider widget with the years using `st.slider`.
import streamlit as st


price1 = st.slider("Minimum Volatility for Heat Map", 0.01, 1.00, 0.01)