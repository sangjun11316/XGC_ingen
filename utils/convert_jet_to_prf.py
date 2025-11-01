import sys, os
import numpy as np

def process_file(input_filepath, factor):
    """
    Reads a data file, extracts the 3rd and 4th columns,
    and writes them to a new .prf file with a header count
    and footer (-1).
    """
    
    # 1. Create the output filename
    # os.path.splitext('data.txt') -> ('data', '.txt')
    # We take the first part [0] and add '.prf'
    base_name = os.path.splitext(input_filepath)[0]
    output_filepath = base_name + '.prf'
    
    print(f"Processing '{input_filepath}'  ->  '{output_filepath}'")

    data_to_write = []

    try:
        with open(input_filepath, 'r') as f:
            # 1. Read and skip the header (first line)
            next(f)
            
            # 2. Read all data lines
            for line in f:
                line_content = line.strip()
                
                # Skip any blank lines
                if not line_content:
                    continue
                    
                # Split the line into columns by whitespace
                parts = line_content.split()
                
                # Make sure the line has enough columns
                if len(parts) >= 4:
                    # Extract the 3rd (index 2) and 4th (index 3) columns
                    col_3 = float(parts[2])
                    col_4 = float(parts[3]) * factor
                    
                    # Store the formatted string
                    data_to_write.append(f"{col_3:20.10e} {col_4:20.10e}")
                else:
                    print(f"Warning: Skipping malformed line: {line_content}")

        # 3. Check if we actually read any data
        if not data_to_write:
            print(f"Error: No data lines found or extracted from '{input_filepath}'.")
            return

        # 4. Write the new .prf file
        with open(output_filepath, 'w') as f_out:
            
            # Write the total number of data rows as the first line
            f_out.write(f"{len(data_to_write)}\n")
            
            # Write all the extracted data
            for data_line in data_to_write:
                f_out.write(f"{data_line}\n")
                
            # Write -1 at the end of the file
            f_out.write("-1\n")
            
        print(f"Successfully created '{output_filepath}' with {len(data_to_write)} data points.")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # 1. Check if the user provided exactly one argument
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} [input_filename] [factor]")
        print(f"Example: python {sys.argv[0]} my_data.txt 1.0")
        sys.exit(1) # Exit with an error code

    # 2. Get input file from the command-line argument
    INPUT_FILE = sys.argv[1] 
    FACTOR = float(sys.argv[2])
 
    # 3. Run the processing function
    process_file(INPUT_FILE, FACTOR)

