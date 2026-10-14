# MMSplice Retraining

mm_splice_retrain_dataset7.ipynb is the original MMSplice training notebook.

The retrain_mmsplice.py python file contains the revised version, with dataset that filters different variant isoforms (with different exon boundaries from the reference). Note that this was made *after* MTSplice retraining, so a lot of functions for building the MMSplice model (particularly with tensorflow 2.17) are changed and imported from mtsplice_retraining.
