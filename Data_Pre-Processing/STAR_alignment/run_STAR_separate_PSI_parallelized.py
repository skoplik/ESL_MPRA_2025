import os
from collections import defaultdict
import sys, getopt
from Bio import SeqIO
import multiprocess as mp  # multiprocessing has issues with SeqIO
import shutil


def create_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir
    

def remove_file(f):
    """ Removes a single file f """
    try:
        os.remove(f)
    except OSError:
        pass


"""
https://github.com/LucksLab/R2D2/blob/master/OSU.py
"""
def remove_files(files):
    """ Remove files in the list files """
    for f in files:
        remove_file(f)


"""
https://github.com/LucksLab/R2D2/blob/master/OSU.py
"""
def run_mapping_pipeline(rec):
    ID=rec.id
    curr_parent_ref = ID.split("_")[0]
    
    # check if current ref is WT
    #if ID_name[0] in wt_supertable_dict.keys() and ID in reference_barcode_dict:
    #if ID in reference_barcode_dict:
    #if ID in reference_barcode_dict and int(curr_parent_ref) <= 244000:
    if close_matches_only and int(curr_parent_ref) <= 244000:
        return False
    elif not close_matches_only and int(curr_parent_ref) > 244000:
        return False
    elif ID in reference_barcode_dict:
        # make FASTA with single reference
        curr_FASTA_file = STAR_indices_dir + ID + ".fa"
        with open(curr_FASTA_file, "w") as single_FA_out:
            SeqIO.write(rec, single_FA_out, 'fasta')
        
        # make STAR index
        genome_dir =  "%s/%s" % (STAR_indices_dir, ID)
        genome_cmd = "STAR --runMode genomeGenerate --genomeSAindexNbases 4 --genomeDir %s --genomeFastaFiles %s --outTmpDir %s_tmp --outFileNamePrefix %s" \
        % (genome_dir, curr_FASTA_file, genome_dir, genome_dir)
        #print(genome_cmd)
        os.system(genome_cmd + " > /dev/null")
                
        # run first pass of STAR
        barcode = reference_barcode_dict[ID]
        curr_STAR_pass1_dir = create_directory(STAR_pass1_dir + "/" + ID + "/")
        R1_fastq = "%s_%s_R1.fastq.gz" % (separate_FASTQ_dir_prefix, barcode)
        R2_fastq = "%s_%s_R2.fastq.gz" % (separate_FASTQ_dir_prefix, barcode)
        if int(curr_parent_ref) <= 244000:
            STAR_pass1_cmd = "STAR --genomeDir %s --readFilesCommand gunzip -c --readFilesIn %s %s --outSAMmultNmax 1 --limitBAMsortRAM 1000290519 \
            --outSAMtype BAM SortedByCoordinate --quantMode GeneCounts --twopassMode Basic --sjdbGTFfile %s --outFileNamePrefix %s/%s --outTmpDir %s_tmp" \
            % (genome_dir, R1_fastq, R2_fastq, gtf_file, curr_STAR_pass1_dir, ID, curr_STAR_pass1_dir)
        else:
            # not including GTF file
            STAR_pass1_cmd = "STAR --genomeDir %s --readFilesCommand gunzip -c --readFilesIn %s %s --outSAMmultNmax 1 --limitBAMsortRAM 1000290519 \
            --outSAMtype BAM SortedByCoordinate --twopassMode Basic --outFileNamePrefix %s/%s --outTmpDir %s_tmp" \
            % (genome_dir, R1_fastq, R2_fastq, curr_STAR_pass1_dir, ID, curr_STAR_pass1_dir)
        #print(STAR_pass1_cmd)
        os.system(STAR_pass1_cmd + " > /dev/null")
        remove_files([curr_FASTA_file])
        
        # process and collapse UMI's
        parsed_dir = create_directory(curr_STAR_pass1_dir + "parsed_alignment/")
        parsed_prefix = parsed_dir + ID
        UMI_fastq = "%s_%s_UMI.fastq.gz" % (separate_FASTQ_dir_prefix, barcode)
                
        filter_cmd = "samtools view -b -q255 -f 0x2 %s/%sAligned.sortedByCoord.out.bam > %s_unique.bam" \
        % (curr_STAR_pass1_dir, ID, parsed_prefix)
        #print(filter_cmd)
        os.system(filter_cmd)  # had issues with /dev/null but will try again
                
        filter_cmd = "java -jar %s CleanSam INPUT=%s_unique.bam OUTPUT=%s_unique_cleaned.bam QUIET=true" \
        % (picard_jar, parsed_prefix, parsed_prefix)
        #print(filter_cmd)
        os.system(filter_cmd + " > /dev/null 2>&1")
                
        filter_cmd = "java -jar %s AnnotateBamWithUmis -i %s_unique_cleaned.bam -f %s -o %s_unique_addUMI.bam --fail-fast" \
        % (fgbio_jar, parsed_prefix, UMI_fastq, parsed_prefix)
        #print(filter_cmd)
        os.system(filter_cmd + " > /dev/null 2>&1")
                
        filter_cmd = "java -jar %s SortBam -i %s_unique_addUMI.bam -o %s_unique_addUMI.sorted.bam --sort-order Queryname" \
        % (fgbio_jar, parsed_prefix, parsed_prefix)
        #print(filter_cmd)
        os.system(filter_cmd + " > /dev/null 2>&1")
        
        filter_cmd = "java -jar %s SetMateInformation -i %s_unique_addUMI.sorted.bam -o %s_unique_addUMI_mate.bam" \
        % (fgbio_jar, parsed_prefix, parsed_prefix)
        #print(filter_cmd)
        os.system(filter_cmd + " > /dev/null 2>&1")
        
        filter_cmd = "java -jar %s GroupReadsByUmi -i %s_unique_addUMI_mate.bam -o %s_unique_UMIcollapse.bam --strategy adjacency -e %s" \
        % (fgbio_jar, parsed_prefix, parsed_prefix, UMI_edit)
        os.system(filter_cmd + " > /dev/null 2>&1")
        
        filter_cmd = "java -jar %s CallMolecularConsensusReads -i %s_unique_UMIcollapse.bam -o %s_unique_UMIconsensus.bam --min-reads 1 --error-rate-post-umi 30" \
        % (fgbio_jar, parsed_prefix, parsed_prefix)
        #print(filter_cmd)
        os.system(filter_cmd + " > /dev/null 2>&1")
        
        filter_cmd = "samtools fastq --threads %s %s_unique_UMIconsensus.bam -1 %s_unique_UMIconsensus_R1.fastq.gz -2 %s_unique_UMIconsensus_R2.fastq.gz" \
        % (processors, parsed_prefix, parsed_prefix, parsed_prefix)
        #print(filter_cmd)
        os.system(filter_cmd + " > /dev/null 2>&1")
        
        # run second pass of STAR
        if int(curr_parent_ref) <= 244000:
            STAR_pass2_cmd = "STAR --genomeDir %s --sjdbGTFfile %s --readFilesCommand gunzip -c \
            --readFilesIn %s_unique_UMIconsensus_R1.fastq.gz %s_unique_UMIconsensus_R2.fastq.gz \
            --outFileNamePrefix %s_unique_UMIconsensus --outSAMmultNmax -1 --limitBAMsortRAM 1000290519 \
            --outSAMtype BAM SortedByCoordinate --quantMode GeneCounts --twopassMode Basic --outTmpDir %s_tmp" \
            % (genome_dir, gtf_file, parsed_prefix, parsed_prefix, parsed_prefix, parsed_prefix)
        else:
            # not giving STAR the GTF file
            STAR_pass2_cmd = "STAR --genomeDir %s --readFilesCommand gunzip -c \
            --readFilesIn %s_unique_UMIconsensus_R1.fastq.gz %s_unique_UMIconsensus_R2.fastq.gz \
            --outFileNamePrefix %s_unique_UMIconsensus --outSAMmultNmax -1 --limitBAMsortRAM 1000290519 \
            --outSAMtype BAM SortedByCoordinate --twopassMode Basic --outTmpDir %s_tmp" \
            % (genome_dir, parsed_prefix, parsed_prefix, parsed_prefix, parsed_prefix)
        #print(STAR_pass2_cmd)
        os.system(STAR_pass2_cmd + " > /dev/null ")
        
        # remove STAR index
        shutil.rmtree(genome_dir)
        
        # clean up first pass
        remove_files([curr_STAR_pass1_dir + ID + "Log.progress.out", curr_STAR_pass1_dir + ID + "ReadsPerGene.out.tab", \
        curr_STAR_pass1_dir + ID + "Log.out"])
        shutil.rmtree(curr_STAR_pass1_dir + ID + "_STARgenome")
        shutil.rmtree(curr_STAR_pass1_dir + ID + "_STARpass1")
        
        # clean up second pass
        remove_files([parsed_prefix + "_unique_addUMI.bai", parsed_prefix + "_unique_addUMI.bam", \
        parsed_prefix + "_unique_addUMI_mate.bam", parsed_prefix + "1_1_unique_addUMI.sorted.bam", \
        parsed_prefix + "_unique_cleaned.bam", parsed_prefix + "_unique_UMIcollapse.bam", \
        parsed_prefix + "_unique_UMIconsensus.bam", parsed_prefix + "_unique_UMIconsensusLog.progress.out",
        parsed_prefix + "_unique_UMIconsensusReadsPerGene.out.tab", parsed_prefix + "_unique_UMIconsensus_R1.fastq.gz", \
        parsed_prefix + "_unique_UMIconsensus_R2.fastq.gz", parsed_prefix + "_unique_UMIconsensusLog.out", \
        parsed_prefix + "_unique.bam", parsed_prefix + "_unique_addUMI.sorted.bam"])
        shutil.rmtree(parsed_prefix + "_unique_UMIconsensus_STARgenome")
        shutil.rmtree(parsed_prefix + "_unique_UMIconsensus_STARpass1")
        return True
    
    return False
    
    
if __name__== "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"", ["GTF_all_refs_file=","all_refs_FASTA_file=", "separate_FASTQ_dir_prefix=", "barcode_ref_file=", \
        "supertable_file=", "p=", "picard_jar=", "fgbio_jar=", "output_dir=", "output_prefix=", "UMI_edit=", "close_matches_only="])
        opts = dict(opts)
        print(opts)
        gtf_file = opts["--GTF_all_refs_file"]
        output_dir = opts["--output_dir"]
        STAR_fasta_file = opts["--all_refs_FASTA_file"]
        separate_FASTQ_dir_prefix = opts["--separate_FASTQ_dir_prefix"]
        barcode_ref_file = opts["--barcode_ref_file"]
        supertable_file = opts["--supertable_file"]
        processors = int(opts["--p"]) if "--p" in opts else 1
        picard_jar = opts["--picard_jar"]
        fgbio_jar = opts["--fgbio_jar"]
        UMI_edit = int(opts["--UMI_edit"]) if "--UMI_edit" in opts else 1
        close_matches_only = opts["--close_matches_only"] == "True" if "--close_matches_only" in opts else False
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path_prefix = "%s/%s" % (output_dir, opts["--output_prefix"])
        
        """
        # For HEK293 Rep 1
        gtf_file = /home/ec2-user/environment/Analysis/clustering_barcodes_DNA/concat_redo_2022_06_02_d1c_ms75_close_match_bowtie/assembled_GTF_ESL_concat_2022_06_02_subsampleparams_d1c_ms75_bowtie_mincov5.gtf
        output_dir = /ESL/Analysis/STAR_alignment/separate
        STAR_fasta_file = /home/ec2-user/environment/Analysis/clustering_barcodes_DNA/concat_redo_2022_06_02_d1c_ms75_close_match_bowtie/ESL_concat_2022_06_02_subsampleparams_d1c_ms75_bowtie_mincov5_reference_with_close_matches.fasta
        separate_FASTQ_dir_prefix = /ESL/Analysis/Filter_RNA-seq_barcodes/filtered_HEK293_Rep1-3_ESL_concat_2022_06_02_subsampleparams_mincov5_close_match_1MM/separate/ESLR1_S1_filtered
        barcode_ref_file = separate_FASTQ_dir + ESLR1_S1_filtered_reference_barcode_counts.txt
        output_path_prefix = HEK293_Rep1_separate
        supertable_file = /home/ec2-user/environment/Data/Sequences/supertable.tsv
        """
    except:
        print("Failed to get opts")
        sys.exit(2)
    
    # collect WT reference numbers, has extras here
    wt_supertable_dict = {}  # supertable reference -> seq_tuple
    wt_eventID_supertableRef_dict = {}  # supertable eventID -> supertable reference
    mut_supertable_dict = defaultdict(list)  # supertable wt reference -> list of mutant reference
    supertable_gene_snp = {}  # supertable reference -> gene + SNP + event_id
    gene_name_dict = {}
    
    with open(supertable_file, "r") as f:
        f.readline()  # throw out header
        for line in f:
            var = line.split("\t")
            curr_supertable_ref = str(int(var[0]) + 1)
            event_id = var[3]
            gene_name = var[6]
            snp = var[11]
            exon = var[4]
            intron1 = var[7]
            intron2 = var[8]
            fullseq_supertable = intron1 + exon + intron2
            seq_tuple = (exon, intron1, intron2)
            # parse WT
            if var[11] == "none":
                wt_supertable_dict[curr_supertable_ref] = seq_tuple
                if event_id not in wt_eventID_supertableRef_dict:
                    wt_eventID_supertableRef_dict[event_id] = curr_supertable_ref
                supertable_gene_snp[curr_supertable_ref] = (gene_name, snp, event_id)
            gene_name_dict[curr_supertable_ref] = gene_name
    print("len(wt_supertable_dict.keys()) = ", len(wt_supertable_dict.keys()))
    
    # get reference to barcode mappings
    reference_barcode_dict = {}  # reference -> barcode
    with open(barcode_ref_file) as f:
        f.readline()  # throw out header
        for line in f:
            var = line.strip().split()
            reference_barcode_dict[var[0]] = var[1]
    
    # run through index generation, mapping, UMI collapse, mapping 
    STAR_indices_dir = create_directory(output_path_prefix + "_STAR_indices/")
    STAR_pass1_dir = create_directory(output_path_prefix + "_STAR_pass1/")
    
    p = mp.Pool(processors)
    
    count = 0
    with open(STAR_fasta_file, "r") as fasta:
        recs = SeqIO.parse(fasta,"fasta")
        for result in p.imap(run_mapping_pipeline, recs, chunksize=1):
            if result:
                count += 1
    
    # load GTF for expected exon junctions
    exon_dict = defaultdict(list)
    exon_start_dict = defaultdict(list)
    exon_end_dict = defaultdict(list)
    transcript_dict = defaultdict(list)
    all_PSIs_cov = {}
    with open(gtf_file, "r") as f:
        curr_expected_junctions = {'i1': [0,0], 'i2': [0,0], 'e': [0,0]}
        prev_parent_ref = -1
        junction_counts = {'i1': 0, 'i2': 0, 'e': 0}
        junction_multi_counts = {'i1': 0, 'i2': 0, 'e': 0}
        for line in f:
            var = line.strip().split("\t")
            # only looking at exons
            if line[0] == "#":
                continue
            if var[2] != "exon":
                continue
            if var[0] not in reference_barcode_dict.keys():
                continue
            
            curr_parent_ref = var[0].split("_")[0]
            
            """
            # only counting WT for now
            if curr_parent_ref not in wt_supertable_dict.keys():
                continue
            """
            
            if prev_parent_ref == -1:
                prev_parent_ref = curr_parent_ref
                # repeat code, just the variable exon part as the pipeline stands
                if var[4] != "1" and var[4] != "872":
                    curr_expected_junctions['i1'][1] = str(int(var[3]) - 1)
                    curr_expected_junctions['i2'][0] = str(int(var[4]) + 1)
            elif curr_parent_ref == prev_parent_ref:
                if var[3] == "1":
                    curr_expected_junctions['i1'][0] = str(int(var[4]) + 1)
                    curr_expected_junctions['e'][0] = str(int(var[4]) + 1)
                elif var[3] == "872":
                    curr_expected_junctions['i2'][1] = str(int(var[3]) - 1)
                    curr_expected_junctions['e'][1] = str(int(var[3]) - 1)
                    # collect reads
                    SJ_file = "%s_STAR_pass1/%s/parsed_alignment/%s_unique_UMIconsensusSJ.out.tab" % (output_path_prefix, var[0], var[0])
                    with open(SJ_file, "r") as sj_f:
                        for line_sj in sj_f:
                            var_sj = line_sj.strip().split()
                            junction = [k for k, v in curr_expected_junctions.items() if v == [var_sj[1], var_sj[2]]]
                            if len(junction) == 1:
                                #print("var[0] = ", var[0])
                                junction_counts[junction[0]] += int(var_sj[6])
                                """
                                print("var_sj = ", var_sj)
                                print("int(var_sj[7]) = ", int(var_sj[7]))
                                """
                                junction_multi_counts[junction[0]] += int(var_sj[7])
                    """
                    print("junction_counts = ", junction_counts)
                    """
                else:
                    curr_expected_junctions['i1'][1] = str(int(var[3]) - 1)
                    curr_expected_junctions['i2'][0] = str(int(var[4]) + 1)
            else:  # moved to new parent ref
                # calculate PSI
                included = min(junction_counts['i1'], junction_counts['i2'])
                total_count = (included + junction_counts['e'])
                """
                print("prev_parent_ref = ", prev_parent_ref)
                print("curr_expected_junctions = ", curr_expected_junctions)
                print("junction_counts = ", junction_counts)
                print("junction_multi_counts = ", junction_multi_counts)
                """
                PSI = included / total_count if total_count > 0 else -1
                
                included_multi = min(junction_counts['i1'] + junction_multi_counts['i1'], junction_counts['i2'] + junction_multi_counts['i2'])
                total_count_multi = (included_multi + junction_counts['e'] + junction_multi_counts['e'])
                multi_PSI = included_multi / total_count_multi if total_count_multi > 0 else -1
                
                all_PSIs_cov[int(prev_parent_ref)] = (PSI, total_count, multi_PSI, total_count_multi)
                
                # reset variables
                prev_parent_ref = curr_parent_ref
                curr_expected_junctions = {'i1': [0,0], 'i2': [0,0], 'e': [0,0]}
                junction_counts = {'i1': 0, 'i2': 0, 'e': 0}
                junction_multi_counts = {'i1': 0, 'i2': 0, 'e': 0}
                
                # repeat code, just the variable exon part as the pipeline stands
                if var[4] != "1" and var[4] != "872":
                    curr_expected_junctions['i1'][1] = str(int(var[3]) - 1)
                    curr_expected_junctions['i2'][0] = str(int(var[4]) + 1)
                
    # Calculate PSI of last reference
    included = min(junction_counts['i1'], junction_counts['i2'])
    total_count = (included + junction_counts['e'])
    PSI = included / total_count if total_count > 0 else -1
    
    included_multi = min(junction_counts['i1'] + junction_multi_counts['i1'], junction_counts['i2'] + junction_multi_counts['i2'])
    total_count_multi = (included_multi + junction_counts['e'] + junction_multi_counts['e'])
    multi_PSI = included_multi / total_count_multi if total_count_multi > 0 else -1
    
    all_PSIs_cov[int(prev_parent_ref)] = (PSI, total_count, multi_PSI, total_count_multi)
    
    # write out PSI's
    with open(output_path_prefix + "_all_PSIs.txt", "w") as f:
        f.write("Reference\tPSI\tCoverage\n")
        for ref in sorted(all_PSIs_cov.keys()):
            #f.write("%s\t%s\t%s\t%s\t%s\n" % (ref, all_PSIs_cov[ref][0], all_PSIs_cov[ref][1], all_PSIs_cov[ref][2], all_PSIs_cov[ref][3]))
            f.write("%s\t%s\t%s\n" % (ref, all_PSIs_cov[ref][0], all_PSIs_cov[ref][1]))
    
    with open(output_path_prefix + "_all_PSIs_mincov10.txt", "w") as f:
        f.write("Reference\tPSI\tCoverage\n")
        for ref in sorted(all_PSIs_cov.keys()):
            if all_PSIs_cov[ref][1] >= 10:
                f.write("%s\t%s\t%s\n" % (ref, all_PSIs_cov[ref][0], all_PSIs_cov[ref][1]))
    
    p.close()
    p.join()
