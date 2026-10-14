import numpy as np


def decode(one_hot_vector):
	index = 0
	for i in one_hot_vector:
		if i > 0:
			return index
		index += 1
	return 4 # all zeros, return N index


def get_decoded_sequences(sequence_one_hots):
	"""
	sequence_one_hots is of shape
	(1, n_sequences, seq_length, 4, 1)
	decode: takes in a (4, 1) one hot vector 
	and turns it into either "A", "C", "G", "T" or "N"
	note: seq is a list of nucleotide strings 
	e.g. ["A", "A", "G", "T", "C", "T"] and joined 
	e.g. to "AAGTCT"
	sequences is a list of strings:
	["AAGTCT", "TTCGTA", ...] each of length seq_length
	"""
	sequences = []
	keys = ["A", "C", "G", "T", "N"]
	squeezed = np.squeeze(sequence_one_hots)
	if len(squeezed.shape) == 2:
		# squeezed the batch dim
		squeezed = np.reshape(squeezed, (1, squeezed.shape[0], squeezed.shape[1]))
	elif len(squeezed.shape) == 4:
		# sample axis exists
		squeezed = np.reshape(squeezed, (squeezed.shape[0] * squeezed.shape[1], \
								   squeezed.shape[2], squeezed.shape[3]))
	for i in range(len(squeezed)):
		# for all sequences in the batch
		seq = []
		for j in squeezed[i]:
			nucleotide = keys[decode(j)]
			seq.append(nucleotide)
		seq = "".join(seq)
		sequences.append(seq)
	return sequences


