import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Get the API token from the environment
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#HUGGINGFACEHUB_API_TOKEN='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'

if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

@st.cache_data
def name_generator(category, subcategory, name, weight, dimensions, short_description):
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                               temperature=0.1, tokens=HUGGINGFACEHUB_API_TOKEN)
    
    if short_description:
        template = """I have a product in the {category} category, specifically a {subcategory}. It's called {name}, weighs {weight}, and has dimensions {dimensions}. Here is a short description: {short_description}. I want a description with five key points about this product. Provide the points only."""
        input_variables = ["category", "subcategory", "name", "weight", "dimensions", "short_description"]
        prompt = PromptTemplate(template=template, input_variables=input_variables)
        response = prompt | llm
        output = response.invoke({
            "category": category,
            "subcategory": subcategory,
            "name": name,
            "weight": weight,
            "dimensions": dimensions,
            "short_description": short_description
        })
    else:
        template = """I have a product in the {category} category, specifically a {subcategory}. It's called {name}, weighs {weight}, and has dimensions {dimensions}. I want a description with five key points about this product. Provide the points only."""
        input_variables = ["category", "subcategory", "name", "weight", "dimensions"]
        prompt = PromptTemplate(template=template, input_variables=input_variables)
        response = prompt | llm
        output = response.invoke({
            "category": category,
            "subcategory": subcategory,
            "name": name,
            "weight": weight,
            "dimensions": dimensions
        })
    
    return output

st.title("Product Description Generator")

category = st.text_input("Product Category:")
subcategory = st.text_input("Subcategory:")
name = st.text_input("Product Name:")
weight = st.text_input("Weight:")
dimensions = st.text_input("Dimensions:")
short_description = st.text_area("Short Description (optional):")

if st.button("Generate Description"):
    response = name_generator(category, subcategory, name, weight, dimensions, short_description)
    st.write(response)
