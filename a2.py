import subprocess
import os

def handle_A2():
    """
    Task A2: Format the contents of /data/format.md using prettier@3.4.2,
    updating the file in-place.
    """
    # Define the path to the file that needs to be formatted.
    file_path = r"C:\Users\Shahzade Alam\Desktop\TDS_P1\data\format.md"
    
    # Check if the file exists before attempting to format it.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Build the command to run prettier.
    # Using npx to run a specific version (prettier@3.4.2) with the --write option.
    command = ["npx", "prettier@3.4.2", "--write", file_path]
    
    try:
        # Run the command. 'check=True' ensures an exception is raised if the command fails.
        subprocess.run(command, check=True)
        return "A2 completed successfully."
    except subprocess.CalledProcessError as e:
        # If the subprocess fails, capture the error and raise an exception.
        raise RuntimeError("Failed to format the file using prettier: " + str(e))
