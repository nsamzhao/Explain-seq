import numpy as np
import modisco
import sys
import os
import pickle
from Bio import SeqIO
import h5py
from collections import OrderedDict
import gzip
import h5py
import modisco.util
import time
import subprocess


# parameters
all_tasks = [0]#, 1, 2, 3, 4]#, 5]
cell_line_names = ['A549']#, 'HCT116', 'HepG2', 'K562', 'MCF-7']#, 'SH-SY5Y']
input_h5py_file = '../data/output_data/starr_sig_A549_499bp_rep1rep2.h5'
central_bp_to_interpret = 499
seq_len = 499
# CHNAGE SEQ FILE NAME!


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


### 1. Read in data ###
task_to_scores = OrderedDict()
task_to_hyp_scores = OrderedDict()
start_bp = int((seq_len - central_bp_to_interpret)/2)
end_bp = int((seq_len + central_bp_to_interpret)/2)

f = h5py.File(input_h5py_file,"r")
tasks = f["contrib_scores"].keys()
for task in tasks:
    #Note that the sequences can be of variable lengths;
    #in this example they all have the same length (200bp) but that is
    #not necessary.
    task_to_scores[task] = [np.array(x)[start_bp: end_bp]
                            for x in f['contrib_scores'][task]]
    task_to_hyp_scores[task] = [np.array(x)[start_bp: end_bp]
                                for x in f['hyp_contrib_scores'][task]]


### 2. Run tf-modisco ###
null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=4900)

for task_idx, cell_line in zip(all_tasks, cell_line_names):
    start_time = time.time()
    
    input_file = f'../data/output_data/starr_sig_valueLabel_499bp_rep1rep2.fa' #CHANGE
    records = list(SeqIO.parse(input_file, "fasta"))
    fasta_sequences = [str(record.seq.upper()) for record in records]
    onehot_data = np.array([one_hot_encode_along_channel_axis(seq)[start_bp: end_bp] for seq in fasta_sequences])

    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                        #Slight modifications from the default settings
                        sliding_window_size=15, #21
                        flank_size=5, #10
                        target_seqlet_fdr=0.15, #because of fdr?
                        max_seqlets_per_metacluster=20000, #50000 in soft paper used 250gb memory
                        seqlets_to_patterns_factory=
                         modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                            #Note: as of version 0.5.6.0, it's possible to use the results of a motif discovery
                            # software like MEME to improve the TF-MoDISco clustering. To use the meme-based
                            # initialization, you would specify the initclusterer_factory as shown in the
                            # commented-out code below:
                            initclusterer_factory=modisco.clusterinit.memeinit.MemeInitClustererFactory(
                                meme_command="meme", base_outdir=f"meme_out/{cell_line}",
                                max_num_seqlets_to_use=10000, nmotifs=20, n_jobs=20, verbose=False),
                            trim_to_window_size=15, #30
                            initial_flank_to_add=5, #10
                            final_min_cluster_size=20,
                            n_cores=20)
                    )(
                     task_names=[f"task{task_idx}"],#, "task1", "task2"],
                     contrib_scores=task_to_scores,
                     hypothetical_contribs=task_to_hyp_scores,
                     one_hot=onehot_data,
                     null_per_pos_scores=null_per_pos_scores)

    ### 3. Save results ###
    output_file = f'../data/output_data/{cell_line}_rep1rep2.h5'
    if (os.path.isfile(output_file)):
        subprocess.run(f"rm {output_file}", shell=True)
    grp = h5py.File(output_file, "w")
    tfmodisco_results.save_hdf5(grp)
    grp.close()
    
    print(f"task{task_idx} takes {time.time() - start_time}", flush=True)
