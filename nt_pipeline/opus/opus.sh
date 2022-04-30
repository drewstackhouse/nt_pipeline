#!/bin/bash
while getopts "s:t:" flag
do
    case "$flag" in
        s) SRC=${OPTARG};;
        t) TGT=${OPTARG};;
    esac
done

shift $(( OPTIND - 1 ))

for corpus in "$@"; do

    # delimit corpus title and version using internal field separator
    IFS="/"
    read -ra PARTS <<< "${corpus}"
    NAME=${PARTS[0]}
    VERSION=${PARTS[1]}

    # create new directory for corpus and download, parse files
    mkdir $NAME
    wget -q -P $NAME https://object.pouta.csc.fi/OPUS-$NAME/$VERSION/moses/$SRC-$TGT.txt.zip
    unzip -q -d $NAME $NAME/$SRC-$TGT.txt.zip
    paste $NAME/$NAME.$SRC-$TGT.$SRC $NAME/$NAME.$SRC-$TGT.$TGT > $(echo $NAME)_$(echo $SRC)_$(echo $TGT).txt
    wc -l $(echo $NAME)_$(echo $SRC)_$(echo $TGT).txt
    rm -r $NAME
    
    echo "----------------------------------------------------"

done


# concat all files where path ends with _${src}_${tgt}.txt
cat *_$(echo $SRC)_$(echo $TGT).txt > $(echo $SRC)_$(echo $TGT).txt

# delete every file besides ${src}_${tgt}.txt
rm -r *_$(echo $SRC)_$(echo $TGT).txt

# output sentence count of combined file
wc -l $(echo $SRC)_$(echo $TGT).txt
