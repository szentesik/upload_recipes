####################################################################################
##
## The script creates chunks from files in a specific directory and uploads  
## them to a remote endpoint for embedding
##
####################################################################################

# Import libraries
import json
from unstructured.partition.md import partition_md
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import (Element, Title, Text, ElementMetadata)
from tqdm import tqdm
from typing import List
import os
from glob import glob
from dotenv import load_dotenv
import re
import requests

print("✅ All libraries imported successfully!")

# take environment variables from .env file
load_dotenv()
print("✅ Test environment loaded!")

data_folder = "./data"
print(f"📂 Loading documents from: {data_folder}")


####################################### load metadata #############################################

file_metadata = { }

# Gather metadata from recipe file
def gather_file_metadata(filename: str):
    data = { 
        "recipe_name": "",        
        "ingredients_count": 0,
        "steps_count": 0,        
        "difficulty": "easy", # "easy|medium|hard", lépések száma alapján
        "prep_time": "N/A"
    }
    try:
        with open(filename, mode="r", encoding="utf-8") as f:
            recipe_name = ""
            ingredients_count = 0
            steps_count = 0
            source_file = os.path.basename(filename)
            prep_time = "N/A"
            while line := f.readline():
                if line.startswith("# "):
                    recipe_name = line[2:]
                    continue
                if line.startswith("## info"):
                    while line := f.readline():
                        if "hour" in line.casefold() or "minute" in line.casefold():      # possible the preparation time                            
                            prep_time = line[2:] if line.startswith("* ") else line
                            break
                        if line.startswith("#"):
                            break   # next section
                if line.startswith("## ingredients"):
                    while line := f.readline():
                        if line.startswith("*"): ingredients_count+=1      # count all bullet list item until next section or EOF
                        if line.startswith("#"):
                            break   # next section
                if line.startswith("## steps"):
                    while line := f.readline():
                        if re.match(r'^\d+\. ', line): steps_count+=1    # count all numbered list item until next section or EOF
                        if line.startswith("#"):
                            break   # next section
            data["recipe_name"] = recipe_name
            data["source_file"] = source_file
            data["ingredients_count"] = ingredients_count
            data["steps_count"] = steps_count
            if steps_count < 5:
                data["difficulty"] = "easy"
            elif steps_count < 9:
                data["difficulty"] = "medium"
            else:
                data["difficulty"] = "hard"
            data["prep_time"] = prep_time
        print(f"➡️ {source_file} contains recipe of {recipe_name}, needs {ingredients_count} ingredients and it takes {steps_count} steps. It is {data["difficulty"]} to make. Preparation time: {prep_time}.")
        return data
    except Exception as e:
        print(f"❌ Unable to open file: {filename}: '{e}'")
        return {}

if not os.path.exists(data_folder):
    raise ValueError(f"Folder {data_folder} not found!")
else:
    # Iterate over the files in the data folder recursively with glob
    for file in tqdm(glob(os.path.join(data_folder, "**"), recursive=True), desc="Loading files"):
        if os.path.isfile(file):
            # read metadata from file
            file_metadata[os.path.basename(file)] = gather_file_metadata(file)  

####################################### Partitioning ##############################################

# Partitioning documents
elements_text = []
elements_md = []

# Reads whole file to a single element
def file_to_element(filename: str) -> Element:
    try:
        with open(filename, mode="r", encoding="utf-8") as f:            
            metadata = ElementMetadata(
                filename = filename                
            )
            metadata.detection_origin = "text"
            if text := f.read():
                element = Text(                    
                    text = text,
                    metadata = metadata
                )
                element.category = "Text"
                return element
    except Exception as e:
        print(f"❌ Unable to process file: {filename}, err: {e}")
        return {}

# Check if folder exists
if not os.path.exists(data_folder):
    raise ValueError(f"Folder {data_folder} not found!")
else:
    # Iterate over the files in the data folder recursively with glob
    for file in tqdm(glob(os.path.join(data_folder, "**"), recursive=True), desc="Loading files"):
        if os.path.isfile(file):            
            # Partition the document
            elements_text.append(file_to_element(filename=file))
            elements_md.extend(partition_md(filename=file))

    # Display the extracted elements
    print(f"✅ Total elements extracted in text: {len(elements_text)}")
    print("\n📋 Element types and content:")
    for i, element in enumerate(elements_text[:10]):  # Show first 10 elements
        print(f"{i+1}. Type: {element.category},\nContent:\n{str(element)}")
        print("-"*40)

    print("="*120)

    print(f"✅ Total elements extracted in md: {len(elements_md)}")
    print("\n📋 Element types and content:")
    for i, element in enumerate(elements_md[:10]):  # Show first 10 elements
        print(f"{i+1}. Type: {element.category},\nContent:\n{str(element)}")
        print("-"*40)

####################################### Chunking ##############################################

# Chunking elements
## Method 2: By title chunking

chunks_by_title = chunk_by_title(
    elements_md,
    max_characters=500,  # Hard maximum for chunk size
    new_after_n_chars=400,
    combine_text_under_n_chars=100,  # Combine small sections
    multipage_sections=False,  # Don't preserve page boundaries
    include_orig_elements=True
)

print(f"By Title Strategy - Total chunks created: {len(chunks_by_title)}")
print("\n" + "="*60)

# Display the chunks with their titles
for i, chunk in enumerate(chunks_by_title[:25]): 
    print(f"\nChunk {i+1}:")
    print(f"Length: {len(str(chunk))} characters")
    
    # Check if chunk has metadata with title information
    if hasattr(chunk, 'metadata'):
        print(f"Metadata: {chunk.metadata.to_dict()}")
        if hasattr(chunk.metadata, 'orig_elements'):            
            print(f"Kind: {chunk.metadata.orig_elements[0]} ({str(type(chunk.metadata.orig_elements[0]))})")
            print(*chunk.metadata.orig_elements, sep='\n')
    
    print(f"Content: {str(chunk)}")
    print("-" * 80)

# Chunking elements
## Method 3: Basic chunking (fix size with overlap)

chunks_basic = chunk_elements(
    elements_text,
    max_characters=500,  # Hard maximum for chunk size
    new_after_n_chars=400,
    overlap=100,    
    include_orig_elements=True,  # Hard maximum for chunk size
)

print(f"Basic Strategy - Total chunks created: {len(chunks_basic)}")
print("\n" + "="*60)

# Display the chunks
for i, chunk in enumerate(chunks_basic[:5]):  # Show first 5 chunks
    print(f"\nChunk {i+1}:")
    print(f"Length: {len(str(chunk))} characters")
    # Check if chunk has metadata with title information
    if hasattr(chunk, 'metadata'):
        print(f"Metadata: {chunk.metadata.to_dict()}")
    print(f"Content: {str(chunk)}")
    print("-" * 80)

##################################### Create documents ############################################

# convert chunks to documents (one document per recipe)
documents_baseline = []
for i, chunk in enumerate(elements_text):    
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'filename'): 
        source_file = chunk.metadata.filename
        try:
            filedata = file_metadata[source_file]
        except KeyError as ke:
           filedata = {            
            "recipe_name": "N/A",
            "ingredients_count": 0,
            "steps_count": 0,
            "prep_time": "N/A",
            "difficulty": "N/A",
           }            
    doc = {
        "id": i + 1,        
        "category": chunk.category if hasattr(chunk, 'category') else "Document",        
        "recipe_name": filedata["recipe_name"],
        "section_type": "full",
        "ingredients_count": filedata["ingredients_count"],
        "steps_count": filedata["steps_count"],
        "prep_time": filedata["prep_time"],
        "difficulty": filedata["difficulty"],
        "source_file": source_file,
        "text": str(chunk),
        "chunking_method": "single file"
    }
    documents_baseline.append(doc)

print(f"✅ Prepared {len(documents_baseline)} chunked documents (baseline)")

# Convert chunks to documents (fragmented by section)
documents_by_title = []
recipe_name = "N/A"
section_type = "N/A"
for i, chunk in enumerate(chunks_by_title):
    # Obtain section type of actual chunk (ingredients|steps|notes|full)
    if hasattr(chunk.metadata, 'orig_elements') and isinstance(chunk.metadata.orig_elements[0], Title):
        kind = str(chunk.metadata.orig_elements[0])
        #print(f"Kind #{i}: {kind}")
        if(kind.endswith(':')):
            kind = kind[:-1]        
        if kind in ["info", "ingredients", "steps", "notes", "based on"]:
            #print(f"Section: {kind}")            
            section_type = kind
        else:
            #print(f"Recipe: {kind}")
            section_type = "title"
            recipe_name = kind
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'filename'): 
        source_file = chunk.metadata.filename
        try:
            filedata = file_metadata[source_file]
        except KeyError as ke:
           filedata = {            
            "recipe_name": recipe_name,
            "ingredients_count": 0,
            "steps_count": 0,
            "prep_time": "N/A",
            "difficulty": "N/A",
           }            
    doc = {
        "id": i + 1,        
        "category": chunk.category if hasattr(chunk, 'category') else "Document",        
        "recipe_name": filedata["recipe_name"],
        "section_type": section_type,  # "ingredients|steps|notes|full"
        "ingredients_count": filedata["ingredients_count"],
        "steps_count": filedata["steps_count"],
        "prep_time": filedata["prep_time"],
        "difficulty": filedata["difficulty"],
        "source_file": source_file,
        "text": str(chunk),
        "chunking_method": "section based"
    }
    documents_by_title.append(doc)

print(f"✅ Prepared {len(documents_by_title)} chunked documents (per section)")

# Convert chunks to documents (fix sized overlapped chunks)
documents_overlapped = []
for i, chunk in enumerate(chunks_basic):    
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'filename'): 
        source_file = chunk.metadata.filename
        try:
            filedata = file_metadata[source_file]
        except KeyError as ke:
           filedata = {            
            "recipe_name": "N/A",
            "ingredients_count": 0,
            "steps_count": 0,
            "prep_time": "N/A",
            "difficulty": "N/A",
           }            
    doc = {
        "id": i + 1,        
        "category": chunk.category if hasattr(chunk, 'category') else "Document",        
        "recipe_name": filedata["recipe_name"],
        "section_type": "full",        
        "ingredients_count": filedata["ingredients_count"],
        "steps_count": filedata["steps_count"],
        "prep_time": filedata["prep_time"],
        "difficulty": filedata["difficulty"],
        "source_file": source_file,
        "text": str(chunk),
        "chunking_method": "fixed size"
    }
    documents_overlapped.append(doc)

print(f"✅ Prepared {len(documents_overlapped)} chunked documents (fixed size overlapped)")


########################################### Upload ################################################

api_endpoint=os.getenv("UPLOAD_API_ENDPOINT", "")
if not api_endpoint:
    print(f"❌ Environment variable UPLOAD_API_ENDPOINT is not defined")
    quit()


def upload_documents(documents: list, url: str):
    uploaded = 0
    for doc in tqdm(documents, desc="Uploading documents"):                        
        res = requests.post(url, json = doc)
        if(res.status_code == 201):
            uploaded +=1
        else:
            print(f"⚠️ Resource not created: {res.status_code}, {res.text}")
    return uploaded

uploaded = upload_documents(documents_baseline, api_endpoint)
print(f"✅ {uploaded} / {len(documents_baseline)} documents uploaded")