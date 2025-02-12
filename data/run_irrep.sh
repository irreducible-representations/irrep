logfile="$(pwd)/log_failed_SGs.dat"
if [ -e $logfile ]; then
    rm $lofgile
    touch $logfile
fi

for i in $(seq 1 230); do
    
    if [ $i -eq 168 ] || [ $i -eq 207 ]; then
        continue
    fi

    dir="sg_$i"
    cd $dir
    irrep -onlysym -spinor -code=fplo -sg=$i


    if [ $? -ne 0 ]; then  # error when in IrRep run
        echo "$i" >> $logfile
    fi
    cd ..

done
