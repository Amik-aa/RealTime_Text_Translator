import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Function to translate text
def translate_text_multilang(text, src_lang, tgt_lang):
   
    # Define supported language pairs (limited to German, Italian, Spanish, and Dutch)
    supported_lang_pairs = [
        ('en', 'de'), ('en', 'it'), ('en', 'es'), ('en', 'nl')
    ]
    
    if (src_lang, tgt_lang) in supported_lang_pairs:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    else:
        return f"Translation pair {src_lang} to {tgt_lang} not directly supported."
    

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Tokenize and generate translation
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Streamlit app layout
st.title("Real-Time Text Translator")
st.markdown("""
Welcome to the Real-Time Text Translator.
Select your source and target languages, and input the text you want to translate.
""")

# Language selection 
languages = {
    'English': 'en', 'German': 'de', 'Italian': 'it', 'Spanish': 'es', 'Dutch': 'nl'
}

# Source language selection
src_lang = st.selectbox("Select Source Language", list(languages.keys()))
src_lang_code = languages[src_lang]

# Target language selection 
tgt_lang = st.selectbox("Select Target Language", list(languages.keys()))
tgt_lang_code = languages[tgt_lang]

# Text input from the user
text_input = st.text_area("Enter the text to translate:")

# Button to trigger translation
if st.button("Translate"):
    if text_input:
        st.write("Translating... Please wait.")
        # Translating the text
        translated_text = translate_text_multilang(text_input, src_lang_code, tgt_lang_code)
        
        # Displaying the translated text
        st.write("### Translated Text:")
        st.write(translated_text)
    else:
        st.write("Please enter some text to translate.")
