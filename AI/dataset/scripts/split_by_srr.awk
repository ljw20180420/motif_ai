#!/usr/bin/awk -f

{
    SRR=$4;
    sub(/_peak_.*$/, "", SRR);
    filename = ENVIRON["DATA_DIR"] "/splited/" accession "/" SRR ".sorted.narrowPeak";
    print $0 > filename;
}
