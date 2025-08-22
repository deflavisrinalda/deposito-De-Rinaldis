import streamlit as st

st.title('Counter Example')

# #esempio NON funzionante
# count = 0

# increment = st.button('Increment')
# if increment:
#     count += 1

# st.write('Count = ', count)

#esempio funzionante
# Inizializza il contatore solo una volta
if 'count' not in st.session_state:
    st.session_state.count = 0

# Bottone per incrementare
if st.button('Increment'):
    st.session_state.count += 1

# Mostra il valore attuale
st.write('Count = ', st.session_state.count)


# Esempio bottone di reset
if st.button('Reset'):
    st.session_state.count = 0
