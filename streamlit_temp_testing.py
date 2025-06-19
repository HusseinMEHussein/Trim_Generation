import streamlit as st

st.title("Container Example")

# Section 1
with st.container():
    st.subheader("Section 1")
    st.write("This section has related content.")
    st.button("Click Me 1")

# Section 2
with st.container():
    st.subheader("Section 2")
    st.write("This is another grouped area.")
    st.button("Click Me 2")

# Section 3
with st.container():
    st.subheader("Section 3")
    st.write("All items here stay together.")
    st.button("Click Me 3")
