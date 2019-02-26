NTASKS=8
TOTRECORDS=47442
BSTART=$((23720+3383))
NRECORDS=$((TOTRECORDS-BSTART))
BSIZE=$((NRECORDS/(NTASKS-1)))

for (( i=0; i < $NTASKS ; i++ ));
    do echo $i python getItemsByPatientConvert.py 200 6 12 24 36 48 72 -o data/bypt/ -n $(((i*BSIZE)+BSTART)) $BSIZE;
done
