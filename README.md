# NT Pipeline

    ! git clone https://github.com/drewstackhouse/nt_pipeline.git
    % cd nt_pipeline
    ! pip install -q -r requirements.txt

## OPUS

    ! bash ./nt_pipeline/opus/opus.sh -s 'en' -t 'ga' 'CCMatrix/v1' 'EUbookshop/v2'

## Tokenizer

    ! python3 ./nt_pipeline/tokenizer/build_vocab.py -i 'tok_en_ga.txt' -s 'src_vocab.txt' -t 'tgt_vocab.txt'
    ! python3 ./nt_pipeline/tokenizer/custom_tokenizer.py -s src_vocab.txt -t tgt_vocab.txt -o src_tgt_converter

## Transformer

## Translator

## Deploy
