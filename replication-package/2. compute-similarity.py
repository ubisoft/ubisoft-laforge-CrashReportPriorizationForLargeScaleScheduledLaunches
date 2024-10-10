from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss
import torch
import os
import pandas as pd
from transformers import file_utils
print(file_utils.default_cache_path)
import subprocess
subprocess.run(["nvidia-smi"]) 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# Function to Compute Similarity
def calculate_similarities_in_batches(g1, g2, g1_embeddings, g2_embeddings, similarity_path, batch_size=2):
    similarities = []
    for i in range(0, g1_embeddings.shape[0], batch_size):
        g1_batch = g1_embeddings[i:i+batch_size]
        for j in range(0, g2_embeddings.shape[0], batch_size):
            file_path = similarity_path+model_name.replace("/","_")+'_embedding_similarities_'+g1+'_'+str(i)+'_'+g2+'_'+str(j)+'.pt'
            print(i,j)
            g2_batch = g2_embeddings[j:j+batch_size]
            batch_similarities = model.similarity(g1_batch, g2_batch)
            torch.save(batch_similarities, file_path)



model_name = "codellama/CodeLlama-7b-hf"
home = 'data/'

g1_text_embed_dir = home + "output/g1-embed/"
g2_text_embed_dir = home + "output/g2-embed/"

g1_embeddings_path = g1_text_embed_dir+"g1_embeddings.pt"
g2_embeddings_path = g1_text_embed_dir+"g2_embeddings.pt"

similarity_path = home+"/output/similarities/

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(model_name, device=device)#.to(device)

# load embeddings
g1_embeddings = torch.load(g1_embeddings_path)
g2_embeddings = torch.load(g2_embeddings_path)

# Compute Similarity
calculate_similarities_in_batches(g1_embeddings, g2_embeddings, similarity_path, batch_size=20000)




g1_g2_text_embed_similar_pairs_dir =  home+"/output/embed-similarity-pairs/"

g1_occurrences = pd.read_csv(g1_text_embed_dir+"g1_callstack_filtered_data.csv")
g2_occurrences = pd.read_csv(g2_text_embed_dir+"g2_callstack_filtered_data.csv")
g1_sentences = g1_occurrences['Text'].values
g2_sentences = g2_occurrences['Text'].values






li_indices = []
li_similarities = []
def get_similarities_in_batches_project(project_g1, project_g2, sentences_1, sentences_2, g1_g2_text_embed_similar_pairs_dir,batch_size=2,threshold = 0.999):
    similarities = []
    for i in range(0, len(sentences_1), batch_size):
        for j in range(0, len(sentences_2), batch_size):
            file_path = g1_g2_text_embed_similar_pairs_dir +model_name.replace("/","_")+'_embedding_similarities_'+project_g1+'_'+str(i)+'_'+project_g2+'_'+str(j)+'.pt'
            print(i,j)
            similarities =  torch.load(file_path)
            
            # Mask for similarities greater than the threshold
            mask = similarities >= threshold
            
            # Get the indices of the elements where the mask is True
            indices = torch.nonzero(mask).cpu().numpy()

            print(indices)

            col_1 = 0
            col_2 = 1

            indices[:, col_1] += i
            indices[:, col_2] += j

            print(indices)
            
            similarities = similarities[similarities >= threshold].cpu().numpy()
            li_indices.append(indices)
            li_similarities.append(similarities)

            df = pd.DataFrame({
                "index": [i_ for i_ in indices],
                "similarity": similarities
            })



            df.to_json(g1_g2_text_embed_similar_pairs_dir+'codellama-similarities_'+project_g1+'_'+str(i)+'_'+project_g2+'_'+str(j)+'.json') 

def get_indices(element, lst):
    indices = []
    for i in range(len(lst)):
        if lst[i] == element:
            indices.append(i)
    return indices
            
get_similarities_in_batches_project("g1",'g2', g1_sentences, g2_sentences, g1_g2_text_embed_similar_pairs_dir, batch_size=20000,threshold = 0.9)



import re

dict_occurrences = {
    "g1": g1_occurrences[['CrashType']],
    "g2": g2_occurrences[['CrashType']]
}


def get_data(read_path,filename, projects, save_path):
    """
    Find occurrences of specific substrings in a sentence.

    Parameters:
    - sentence (str): The sentence to search within.
    - substrings (list): A list of substrings to find in the sentence.

    Returns:
    - dict: A dictionary where keys are substrings and values are lists of tuples with start and end positions.
    """

    save_file = save_path + filename
    # print("save_file",save_file)
    # print("filename",filename)
    # print(save_file)
    if len(projects)==2:

        # if (os.path.exists(save_file)):
        #     return None
        # Initialize a dictionary to store occurrences
        occurrences = {substring: [] for substring in projects}
    
        # Search for each substring in the sentence
        for substring in projects:
            for match in re.finditer(re.escape(substring), filename):
                occurrences[substring].append(match.start())

        df = pd.read_json(read_path+filename)
    
        if len(occurrences[projects[0]])==2:
            project1 = projects[0]
            project2 = projects[0]
        elif len(occurrences[projects[1]])==2:
            project1 = projects[1]
            project2 = projects[1]
        else:
            if occurrences[projects[0]][0]>occurrences[projects[1]][0]:
                project1 = projects[1]
                project2 = projects[0]
            else:
                project1 = projects[0]
                project2 = projects[1]

        # print("project1",project1)
        # print("project2",project2)

        indices = df['index']

        #print(indices)
        
        row_project1_indices = [i[0] for i in indices]
        col_project2_indices = [i[1] for i in indices]
        similarities = df['similarity']

        # print("row_project1_indices",row_project1_indices)
        # print("col_project2_indices",col_project2_indices)
        
        try:
            # Select records corresponding to each index in the tensor
            selected_rows = dict_occurrences[project1].iloc[row_project1_indices].reset_index(drop=True)
            selected_cols = dict_occurrences[project2].iloc[col_project2_indices].reset_index(drop=True)
        
        except:
            print(filename, len(dict_occurrences[project2]))
            remove_positions = get_indices(len(dict_occurrences[project2]),col_project2_indices)
            print(remove_positions)
            
            for i_ in remove_positions:
                if i_ < len(col_project2_indices):
                    del col_project2_indices[i_]
                    del row_project1_indices[i_]
                if i_ < len(row_project1_indices):
                    del col_project2_indices[i_]
                    del row_project1_indices[i_]
            
            # Select records corresponding to each index in the tensor
            selected_rows = []
            selected_cols = []
            
            for j in range(len(row_project1_indices)):
                selected_row = None
                selected_col = None
                if  (row_project1_indices[j]<len(dict_occurrences[project1])) & (col_project2_indices[j]<len(dict_occurrences[project2])):
                    
                    # print(j, len(dict_occurrences[project1].reset_index()), len(dict_occurrences[project2].reset_index()))
                    
                    selected_row = dict_occurrences[project1].iloc[[row_project1_indices[j]]].reset_index(drop=True)
                    # print(project1,"rows done")
                    # print("col_project2_indices[j]",col_project2_indices[j], col_project2_indices[j]>=len(dict_occurrences[project2]))
                    selected_col = dict_occurrences[project2].iloc[[col_project2_indices[j]]].reset_index(drop=True)
                    # print(project2,"columns done")

                    selected_rows.append(selected_row)
                    selected_cols.append(selected_col)
                # else:
                #     print("else", row_project1_indices[j]<len(dict_occurrences[project1]),  col_project2_indices[j]<len(dict_occurrences[project2]))

            selected_rows = pd.concat(selected_rows).reset_index(drop=True)
            selected_cols = pd.concat(selected_cols).reset_index(drop=True)

            

                    
    
        
        # Combine the selected rows and columns into a single DataFrame
        selected_records = pd.concat([selected_rows, selected_cols], axis=1)
        selected_records['similarity'] = similarities
        selected_records.columns = ['Project1CrashType', 'Project2CrashType', 'similarity']

        selected_records['Project1'] = project1
        selected_records['Project2'] = project2

        selected_records.to_json(save_file)
        print(save_file, "saved.")

        
    
        return selected_records
    else:
        raise ValueError("Two projects were not provided")


projects = ["g1", "g2"]
g1_g2_text_embed_similar_pairs_processed_dir = home+"/output/embed-similarity-pairs-processed/"        
for filename in os.listdir(g1_g2_text_embed_similar_pairs_dir):
    get_data(g1_g2_text_embed_similar_pairs_dir,filename, projects, g1_g2_text_embed_similar_pairs_processed_dir)


acc_data = []

li = os.listdir(g1_g2_text_embed_similar_pairs_processed_dir)
li.sort()

for filename in li:
    try:
        print(filename)
        data = pd.read_json(g1_g2_text_embed_similar_pairs_processed_dir+filename)
        data = data[data['Project1JiraKey']!=data['Project2JiraKey']]
        data = data[data['similarity']>0.90]
        data = data[['Project1JiraKey','Project2JiraKey']].values.tolist()
        
        acc_data+= data
    except Exception as e:
        print("ERROR",filename, e)

import json
with open(g1_g2_text_embed_similar_pairs_processed_dir+'merged_data_0.9_updated.json', 'w') as f:
    json.dump(acc_data, f)


li = os.listdir(g1_g2_text_embed_similar_pairs_processed_dir)
li.sort()

for filename in li:
    try:
        data = pd.read_json(g1_g2_text_embed_similar_pairs_processed_dir+filename)
        data = data[data['Project1JiraKey']!=data['Project2JiraKey']]
        data = data[data['similarity']>0.99]
        data = data[['Project1JiraKey','Project2JiraKey']] 
        acc_data.append(data)
    except Exception as e:
        print("ERROR",filename, e)

acc_data = pd.concat(acc_data)
acc_data.to_json(g1_g2_text_embed_similar_pairs_processed_dir+'merged_data_0.99_updated.json')

