for i in $(seq 1 230); do
    
    dir="sg_$"i
    if [ ! -d $dir ]; then
        mkdir $dir
    fi

    python cif2in.py $i
    cp "=.in" $dir

done
