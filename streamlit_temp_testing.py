
import streamlit as st

toggle = st.toggle("Enable feature")

if toggle:
    st.write("Feature is ON")
else:
    st.write("Feature is OFF")