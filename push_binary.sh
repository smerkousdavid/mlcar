#!/bin/bash

echo "Pushing the mlcapture executable..."

scp -r ~/ML/mlcar/bin/mlcapture mlcar:/home/udooer/mlcapture/bin/mlcapture

echo "Done!"
