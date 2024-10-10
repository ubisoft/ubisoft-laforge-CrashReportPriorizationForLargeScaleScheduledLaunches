import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss
import torch
import os
import pandas as pd
from transformers import file_utils
import re
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def preprocess (frame):
    # include the pre-processing steps such as
    # remove the reporter related frames
    return frame
    
# Function to encode sentences in batches
def encode_sentences_in_batches(project,sentences, model, text_embed_dir,batch_size=2):
    embeddings = []
    uffix = model_name.replace("/","_")+'_'+project+'_batch_embeddings'
    for i in range(13600, len(sentences), batch_size):
        print("batch",i)
        file_path = text_embed_dir+suffix+str(i)+'.pt'
        if os.path.exists(file_path)==False:
            batch = sentences[i:i+batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=True)
            torch.save(batch_embeddings, file_path)
            del batch_embeddings
    
    # List all .pt files in the text_embed_dir
    pt_files = [f for f in os.listdir(text_embed_dir) if f.endswith('.pt')]
    # Create a dictionary to store the contents of all .pt files
    combined_data = {}
    s
    # Load each .pt file and add its contents to the combined_data dictionary
    for pt_file in pt_files:
        if project+"_embeddings.pt" in pt_file:
            continue
        elif "callstack_filtered_data" in pt_file:
            continue
        file_path = os.path.join(text_embed_dir, pt_file)
        data = torch.load(file_path)
        combined_data[int(pt_file.replace(suffix, '').replace('.pt', ''))] = data
    combined_data = dict(sorted(combined_data.items()))
    tensor_list = list(combined_data.values())
    flattened_tensor_list = [item.cpu() for sublist in tensor_list for item in sublist]
    tensor_flattened_tensor_list = torch.tensor(flattened_tensor_list)
    torch.save(tensor_flattened_tensor_list, text_embed_dir+project+"_embeddings.pt")

# Function to encode sentences of a project
def encode(model, project, callstack_path, text_embed_dir, batch_size):
    occurrences = pd.read_csv(callstack_path)
    CrashTypes_report_entries = occurrences[['CrashType','ReportId']].drop_duplicates()
    CrashTypes_report_entries = CrashTypes_report_entries.sort_values('ReportId')
    CrashTypes_report_entries = CrashTypes_report_entries.groupby('CrashType').first().reset_index()
    occurrences = CrashTypes_report_entries.merge(occurrences, on=['CrashType','ReportId'])
    occurrences = occurrences.dropna(subset=["FunctionName"])
    occurrences = occurrences.sort_values(["ReportEntryId","Order"]).groupby("ReportId").head(3)
    occurrences['FunctionName'] = occurrences['FunctionName'].map(preprocess)
    occurrences.to_csv(text_embed_dir+project+"_callstack_filtered_data.csv")
    sentences = occurrences['FunctionName'].values
    encode_sentences_in_batches(project,sentences, model, text_embed_dir, batch_size=batch_size)

home = 'data/'
g1_callstack_path = home+'g1-callstacks.csv'
g2_callstack_path = home+'g2-callstacks.csv'
g1_text_embed_dir = home + "output/g1-embed/"
g2_text_embed_dir = home + "output/g2-embed/"
model_name = "codellama/CodeLlama-7b-hf"

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = SentenceTransformer(model_name, device=device)
if model.tokenizer.pad_token == None: model.tokenizer.pad_token = model.tokenizer.eos_token
encode(model, "g1", g1_callstack_path, g1_text_embed_dir, 20000)
encode(model, "g2", g2_callstack_path, g2_text_embed_dir, 20000)




