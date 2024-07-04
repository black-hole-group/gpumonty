#!/bin/bash

# Loop to execute the command 30 times
for i in $(seq 10 20); do
    # Construct the command with the current value of i
    cmd="time ./grmonty 500000 ./data/dump019 gpumonty_$i"
    
    # Execute the command
    echo "Executing: $cmd"
    $cmd
done
