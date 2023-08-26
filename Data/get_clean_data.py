import h5py
import numpy as np

def get_spiketimes_list(file):

    data = file['spiketimes']

    # Get the number of elements in the dataset
    num_elements = data.shape[0]
    
    # Iterate through the elements and access the data
    spiketimes_list = []
    for i in range(num_elements):
        element = data[i][0]
        if isinstance(element, h5py.Reference):
            # Dereference the object and store it as a numpy array
            spiketimes_list.append(np.array(file[element]).squeeze().tolist())
        else:
            raise TypeError("Element is not a reference")
    return spiketimes_list

def get_stim_onsets_list(file):

    data = file['stim_onsets']

    # Get the number of elements in the dataset
    num_elements = data.shape[0]
    
    # Iterate through the elements and access the data
    stim_onsets_list = []
    for i in range(num_elements):
        element = data[i]
        if isinstance(element, h5py.Reference):
            # Dereference the object and store it as a numpy array
            stim_onsets_list.append(np.array(file[element]).squeeze().tolist())
        else:
            raise TypeError("Element is not a reference")
    return stim_onsets_list

def get_situations_list(file):

    data = file['situations']

    # Get the number of elements in the dataset
    num_elements = data.shape[0]
    
    # Iterate through the elements and access the data
    situations_list = []
    for i in range(num_elements):
        element = data[i][0]
        if isinstance(element, h5py.Reference):
            # Dereference the object and store it as a numpy array
            situations_list.append(np.array(file[element]).squeeze().tolist()[1])
        else:
            raise TypeError("Element is not a reference")
    return situations_list

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

def get_clean_data(path):

    path = os.path.join(current_dir, 'CleanData', path)

    with h5py.File(path, 'r') as file:
        spiketimes_list = get_spiketimes_list(file)
        stim_onsets_list = get_stim_onsets_list(file)
        situations_list = get_situations_list(file)
    
    return spiketimes_list, stim_onsets_list, situations_list

if __name__ == "__main__":

    spiketimes_list, stim_onsets_list, situations_list = get_clean_data(path="w065-0359.jld2")

    print()
    print(f"len(spiketimes_list) = {len(spiketimes_list)}")
    print(f"len(stim_onsets_list) = {len(stim_onsets_list)}")
    print(f"len(situations_list) = {len(situations_list)}")

    print()
    print(f"spiketimes_list[0] = {spiketimes_list[0]}")
    print(f"len(spiketimes_list[0]) = {len(spiketimes_list[0])}")
    print(f"stim_onsets_list[0] = {stim_onsets_list[0]}")
    print(f"situations_list[0] = {situations_list[0]}")

    print()
    print(f"spiketimes_list[1] = {spiketimes_list[1]}")
    print(f"len(spiketimes_list[1]) = {len(spiketimes_list[1])}")
    print(f"stim_onsets_list[1] = {stim_onsets_list[1]}")
    print(f"situations_list[1] = {situations_list[1]}")
