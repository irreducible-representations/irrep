#!/bin/bash
shopt -s expand_aliases
alias fplo23="/home/mi2/software/23.00-65/FPLO/bin/fplo23.00-65-x86_64"

for i in $(seq 1 230); do

    cp '=.groupoutput' $dir
    dir="sg_$i"
    cd $dir
    fplo23
    cd ..

done
