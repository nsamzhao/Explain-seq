"""Transfer weights from begula_var_input to checkpoint. Not include fully connected layers.
"""
import torch
from torch import nn
import argparse

def transfer_beluga(checkpoint_path, beluga_path):
    checkpoint = torch.load(checkpoint_path)
    beluga_dict = torch.load(beluga_path)
    assert len(checkpoint['state_dict'].keys()) == len(beluga_dict.keys())
    
    for checkpoint_state, beluga_state in zip(checkpoint['state_dict'].keys(), beluga_dict.keys()):
        if beluga_state not in ['model.1.2.1.weight', 'model.1.2.1.bias', 'model.1.4.1.weight', 'model.1.4.1.bias']:
            checkpoint['state_dict'][checkpoint_state] = beluga_dict[beluga_state]
    return checkpoint

parser = argparse.ArgumentParser(description='Transfer weights from our base model.')
parser.add_argument('checkpoint_file', type=str, help='Checkpoint file.')
args = parser.parse_args()

checkpoint_transfered = transfer_beluga(checkpoint_path=args.checkpoint_file, beluga_path='/data/data_repo/samzhao/transfer_learn_deepsea/beluga_model/deepsea.beluga.pth')
#'/data/data_repo/samzhao/transfer_learn_deepsea/gm12878/training_outputs/initial_train/best_model.pth.tar'

torch.save(checkpoint_transfered, './transfered_beluga.pth.tar')

