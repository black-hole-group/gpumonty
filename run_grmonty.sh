#!/bin/bash

# Loop to run the command 20 times
for i in {1..20}; do
    # Define the output file name
    output_file="/SANE_${i}.spec"
    
    # Print the current run information
    echo "BEGINNING ${i} OUT OF 20 RUNS:"

    # Execute the command
    ./gpumonty 1500000 ./data/SANE_0.9.bin  "$output_file"
    
    # Optionally, print a separator line
    echo "Finished run ${i}, output saved to ${output_file}."
done
