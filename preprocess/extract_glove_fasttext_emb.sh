#!/bin/bash

python3 extract_glove.py
python3 extract_fasttext.py

echo "The files will be stored in a file called data/static_train.pkl data/static_dev.pkl and static_test.pkl"