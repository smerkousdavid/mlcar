#!/bin/bash

echo "Running the mlcapture script on the rc-car"

ssh -t root@mlcar /home/udooer/mlcapture/bin/mlcapture

echo "Done!"
