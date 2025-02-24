#!/bin/bash

fileindices=$1

while read -r i; do
    echo "SG $i"

    dir="sg_$i"
    if [ ! -d $dir ]; then
        mkdir $dir
    fi
    python cif2in.py $i
    cp "=.in" $dir
done < $fileindices
