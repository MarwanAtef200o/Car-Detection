#!/bin/bash

echo "Insert 'img' for Image or 'vid' for video:"
read mode
echo "Insert the path of the file:"
read target

python phase2.py $mode $target