#!/bin/sh

DEFENSE_INDEXES="0 1 2 3 4 5 6"

# PARAM_INDEX  is the index of the  attack trace
# PARAM_INDEXES="0 1 2 3 4 5 6" 
PARAM_INDEXES="2" 

for INDEX in $DEFENSE_INDEXES;
do
    for PARAM_INDEX in $PARAM_INDEXES;
    do
        outfile="out/method"$INDEX"_pindex_"$PARAM_INDEX".txt"
        nohup time python3 -u SPM_with_real_param.py $INDEX $PARAM_INDEX > $outfile 2>&1 &
    done
done

echo "Finished launching spm attack"

# 'wc_1_500ms': 0
# 'wc_1_1s': 1
# 'wc_1_5s': 2
# 'wc_1_10s': 3
# 'wc_4_1s': 4
# 'wc_4_5s': 5
# 'wc_8_20s': 6

