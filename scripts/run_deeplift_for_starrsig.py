import numpy as np
import deeplift
from keras.models import model_from_json
import deeplift.conversion.kerasapi_conversion as kc
from collections import OrderedDict
from deeplift.util import compile_func
from deeplift.util import get_shuffle_seq_ref_function
from deeplift.dinuc_shuffle import dinuc_shuffle #function to do a dinucleotide shuffle
from deeplift.util import get_hypothetical_contribs_func_onehot
from modisco.visualization import viz_sequence
from Bio import SeqIO
import pandas as pd
from tensorflow import keras
import h5py
import os
import time
import pickle
import subprocess


# parameters
all_tasks = [0]#, 1, 2, 3, 4]#, 5]
cell_line_names = ['A549']#, 'HCT116', 'HepG2', 'K562', 'MCF-7']#, 'SH-SY5Y']
model_file = './converted_models/starr_sig_A549_499bp_rep1rep2.h5'
num_refs_per_seq = 100
batch_size = 32
output_file = '../data/output_data/starr_sig_A549_499bp_rep1rep2.h5'
input_file = f'../data/output_data/starr_sig_valueLabel_499bp_rep1rep2.fa' #CHANGE
#CHANGE FASTA FILES!


### 0. Functions for one-hot encoding sequences ###

def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return

def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1):
        assert zeros_array.shape[0] == len(sequence)
    #will mutate zeros_array
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue #leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        if (one_hot_axis==0):
            zeros_array[char_idx,i] = 1
        elif (one_hot_axis==1):
            zeros_array[i,char_idx] = 1


### 1. Prepare deeplift model ###
deeplift_model = kc.convert_model_from_saved_files(model_file)
                                     
multipliers_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0,
                                                              target_layer_idx=-2)
hypothetical_contribs_func = get_hypothetical_contribs_func_onehot(multipliers_func)

#Once again, we rely on multiple shuffled references
hypothetical_contribs_many_refs_func = get_shuffle_seq_ref_function(
    score_computation_function=hypothetical_contribs_func,
    shuffle_func=dinuc_shuffle,
    one_hot_func=lambda x: np.array([one_hot_encode_along_channel_axis(seq)
                                     for seq in x]))
                                     

### 2. Compute scores ###
task_to_contrib_scores = {}
task_to_hyp_contrib_scores = {}

# read in input data
records = list(SeqIO.parse(input_file, "fasta"))
fasta_sequences = [str(record.seq.upper()) for record in records]
onehot_data = np.array([one_hot_encode_along_channel_axis(seq) for seq in fasta_sequences])

for task_idx, cell_line in zip(all_tasks, cell_line_names):
    print(f"On task {task_idx}")
    start_time = time.time()
    
    # for testing
#    start_index = 0; end_index=100
#    onehot_data = onehot_data[start_index:end_index]
#    fasta_sequences = fasta_sequences[start_index:end_index]
    
    #calculate hypothetical scores
    task_to_hyp_contrib_scores[task_idx] =\
         hypothetical_contribs_many_refs_func(
             task_idx=task_idx,
             input_data_sequences=fasta_sequences, #change
             num_refs_per_seq=num_refs_per_seq,
             batch_size=batch_size,
             progress_update=4000,
         )
    #calculate contribution scores
    task_to_contrib_scores[task_idx] = task_to_hyp_contrib_scores[task_idx]*onehot_data #change
    
    #mean normalize hypothetical scores
    task_to_hyp_contrib_scores[task_idx] =\
        (task_to_hyp_contrib_scores[task_idx]-
         np.mean(task_to_hyp_contrib_scores[task_idx],axis=-1)[:,:,None])
    
    print(time.time() - start_time, flush=True)


if (os.path.isfile(output_file)):
    subprocess.run(f"rm {output_file}", shell=True)
f = h5py.File(output_file)
g = f.create_group("contrib_scores")
for task_idx in all_tasks:
    g.create_dataset("task"+str(task_idx), data=task_to_contrib_scores[task_idx])
g = f.create_group("hyp_contrib_scores")
for task_idx in all_tasks:
    g.create_dataset("task"+str(task_idx), data=task_to_hyp_contrib_scores[task_idx])
f.close()












