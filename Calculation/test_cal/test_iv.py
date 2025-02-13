
import os

# Import the functions from your script (if in a module)
from max_cd import process_iv_calculations  # Replace 'your_script' with your actual script name

# Set the path to a test IV file
test_file = os.path.join(os.getcwd(), "test_IV.CSV")

# Run the IV processing function
process_iv_calculations(test_file)