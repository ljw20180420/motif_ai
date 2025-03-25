#!/bin/bash

get_protein_info() {
    PROTEIN_GENE=$1
    accession=$(
        esearch -db protein -query "$PROTEIN_GENE [GENE] AND Mus musculus [ORGN]" |
        efetch -format acc |
        head -n1
    )
    esearch -db protein -query "$accession [ACCN]" |
    efetch -format gpc -mode xml
}

export -f get_protein_info