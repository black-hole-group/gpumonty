#!/bin/bash

# Loop to run the command 20 times
for i in {4..8}; do
    # Calculate 10 raised to the power of i using bc
    power_value=$(echo "10^$i" | bc)
    
    # Define the output file name
   
    
    # Print the current run information
    echo "BEGINNING ${i} OUT OF 8 RUNS:"

    # Execute the command with the correct exponentiation result
    ./gpumonty "$power_value" ./data/SPHERE_TEST_THIN "thin_${i}.spec"
    
    # Optionally, print a separator line
    echo "Finished run ${i}, output saved to thin_${i}.spec"
done

