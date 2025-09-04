#!/bin/bash

# The following chunk was done on small instance
GitHub_path=/home/ec2-user/environment/GitHub/ESL
output_dir=/home/ec2-user/environment/Analysis/gnomAD/output_dbSNP155_Gencode26

gtf_file=/home/ec2-user/environment/Data/Sequences/Gencode_v26/gencode.v26.annotation.gff3
supertable_file=/home/ec2-user/environment/Data/Sequences/supertable.tsv

output_prefix=supertable_dbSNP155_Gencode_v26_overlap_precalc

# Assumes this is run first: /Analysis/gnomAD/run_find_supertable_overlap_gnomAD_UCSC_Gencode26.sh

:<<'END'
# Doing this so I don't have to parse and download full dbSNP155 dataset. REST API also does not work for this track.
cd /home/ec2-user/environment/Analysis/gnomAD/output_Gencode26
grep -P "^161" supertable_Gencode_v26_overlap_BLAT_fullseq_temp.psl | cut -f14,16,17 | grep -P "^chr" > supertable_Gencode_v26_overlap_BLAT_fullseq_match_locs.txt
split -l 1000 supertable_Gencode_v26_overlap_BLAT_fullseq_match_locs.txt
# split makes these files: xaa, xab, xac

# uploaded to UCSC table browser which only takes 1000 locations at a time. 
# Moved results to instance. Going to concat the three parts together:
cd /home/ec2-user/environment/Analysis/gnomAD/dbSNP155_Gencode26
cat supertable_Gencode_v26_overlap_BLAT_fullseq_match_locs_xaa.tsv supertable_Gencode_v26_overlap_BLAT_fullseq_match_locs_xab.tsv supertable_Gencode_v26_overlap_BLAT_fullseq_match_locs_xac.tsv > supertable_Gencode_v26_overlap_BLAT_fullseq_match_locs.tsv
sort supertable_Gencode_v26_overlap_BLAT_fullseq_match_locs.tsv  | uniq > supertable_Gencode_v26_overlap_BLAT_fullseq_match_locs_unique.tsv
# use VIM to fix header which is buried within the file due to sorting
END

dbSNP_table_file=/home/ec2-user/environment/Analysis/gnomAD/dbSNP155_Gencode26/supertable_Gencode_v26_overlap_BLAT_fullseq_match_locs_unique.tsv
BLAT_psl_file=/home/ec2-user/environment/Analysis/gnomAD/output_Gencode26/supertable_Gencode_v26_overlap_BLAT_fullseq_temp.psl

:<<'END'
python3 $GitHub_path/find_supertable_overlap_dbSNP155_UCSC.py --gtf_file $gtf_file --supertable_file $supertable_file \
--output_prefix $output_prefix --output_dir $output_dir --dbSNP_table_file $dbSNP_table_file --BLAT_psl_file $BLAT_psl_file
END




# The following is the redo on the big instance with updated PSI's
GitHub_path=/ESL/GitHub/ESL
supertable_file=/ESL/Data/Sequences/supertable.tsv

STAR_PSI_dir=/ESL/Analysis/STAR_alignment/separate_concat_2023_09_19_d1c_ms75_from_s3_2023_11_27/recount_SJs/
HEK293_Rep1_PSI=$STAR_PSI_dir/HEK293_Rep1_separate_splicing_profiles/HEK293_Rep1_separate_recalc_PSIs_mincov10.txt
HEK293_Rep2_PSI=$STAR_PSI_dir/HEK293_Rep2_separate_splicing_profiles/HEK293_Rep2_separate_recalc_PSIs_mincov10.txt
HEK293_Rep3_PSI=$STAR_PSI_dir/HEK293_Rep3_separate_splicing_profiles/HEK293_Rep3_separate_recalc_PSIs_mincov10.txt

HEK293_WT_Rep1=/ESL/Analysis/WT_Library/separate/recount_SJs/HEK_WT_Rep1_20231101_separate_splicing_profiles/HEK_WT_Rep1_20231101_separate_recalc_PSIs_mincov10.txt
HEK293_WT_Rep2=/ESL/Analysis/WT_Library/separate/recount_SJs/HEK_WT_Rep2_20231101_separate_splicing_profiles/HEK_WT_Rep2_20231101_separate_recalc_PSIs_mincov10.txt

HeLa_Rep2_PSI=$STAR_PSI_dir/HeLa_Rep1_separate_splicing_profiles/HeLa_Rep1_separate_recalc_PSIs_mincov10.txt
HeLa_Rep3_PSI=$STAR_PSI_dir/HeLa_Rep3_20231031_separate_splicing_profiles/HeLa_Rep3_20231031_separate_recalc_PSIs_mincov10.txt

K562_Rep2_PSI=$STAR_PSI_dir/K562_Rep2_separate_splicing_profiles/K562_Rep2_separate_recalc_PSIs_mincov10.txt
K562_Rep3_PSI=$STAR_PSI_dir/K562_Rep3_20231031_separate_splicing_profiles/K562_Rep3_20231031_separate_recalc_PSIs_mincov10.txt

MCF7_Rep1_PSI=$STAR_PSI_dir/MCF7_Rep1_separate_splicing_profiles/MCF7_Rep1_separate_recalc_PSIs_mincov10.txt
MCF7_Rep2_PSI=$STAR_PSI_dir/MCF7_Rep2_20231031_separate_splicing_profiles/MCF7_Rep2_20231031_separate_recalc_PSIs_mincov10.txt


# Assemble all found variants across these studies
dbSNP_notes_supertable_file=/ESL/Figures/Population_variants/Resources/supertable_dbSNP155_Gencode_v26_overlap_dbSNP155_matches_all_notes.txt
#cut -f15,22 /home/ec2-user/environment/Analysis/gnomAD/output_dbSNP155_Gencode26/supertable_dbSNP155_Gencode_v26_overlap_dbSNP155_matches_*.txt | grep -v ucscNotes | sort | uniq > $dbSNP_notes_supertable_file

PSI_file=/ESL/Figures/Variant_analyses/ClinVar/output_collapseRefEvents_psuedocount/HEK293_Reps_2023_09_WT_variants_pseudocount_matched_WT_mutant_PSIs.csv
dPSI_file=/ESL/Figures/Variant_analyses/ClinVar/output_collapseRefEvents_psuedocount/HEK293_Reps_2023_09_WT_variants_pseudocount_HEK293_deltaPSIs.csv
log2A_file=/ESL/Figures/Variant_analyses/ClinVar/output_collapseRefEvents_psuedocount/HEK293_Reps_2023_09_WT_variants_pseudocount_HEK293_log2As.csv
delta_logit_PSI=/ESL/Figures/Variant_analyses/ClinVar/output_collapseRefEvents_psuedocount/HEK293_Reps_2023_09_WT_variants_pseudocount_HEK293_delta_logit_PSIs.csv

output_dir=/ESL/Figures/Population_variants/figures_pseudocount/
output_prefix=supertable_dbSNP155_Gencode_v26_overlap_precalc_psuedocount


python3 $GitHub_path/plot_dPSI_dbSNP155_study_labels_precalc.py --dPSI_file $dPSI_file \
    --dbSNP_notes_supertable_file $dbSNP_notes_supertable_file \
    --output_dir $output_dir --output_prefix $output_prefix \
    --supertable_file $supertable_file --PSI_file $PSI_file --log2A_file $log2A_file --delta_logit_PSI $delta_logit_PSI


# plot AF for each study
studies_order=("1000_genomes" "dbGaP_PopFreq" "TOPMED" "KOREAN" "SGDP_PRJ" "Qatari" "NorthernSweden" "Siberian" "TWINSUK" "TOMMO" \
    "ALSPAC" "GENOME_DK" "GnomAD" "GoNL" "Estonian" "Vietnamese" "Korea1K" "HapMap" "PRJEB36033" "HGDP_Stanford" "Daghestan" \
    "PAGE_STUDY" "Chileans" "MGP" "PRJEB37584" "GoESP" "ExAC" "GnomAD_exomes" "FINRISK" "PharmGKB" "PRJEB37766")

dbSNP_study_supertable_file_prefix=/ESL/Figures/Resources/supertable_dbSNP155_Gencode_v26/supertable_dbSNP155_Gencode_v26_overlap_dbSNP155_matches
study_index=0

output_dir=/ESL/Figures/Population_variants/figures_pseudocount/dbSNP155_Gencode_v26_AF/

for study_name in ${studies_order[@]};
do
    python3 $GitHub_path/plot_dPSI_dbSNP155_study_AF.py --dPSI_file $dPSI_file \
    --dbSNP_study_supertable_file ${dbSNP_study_supertable_file_prefix}_${study_name}.txt \
    --output_dir $output_dir --output_prefix $output_prefix --study_index $study_index --study_name $study_name
    
    ((study_index++))  # increment study_index
done

