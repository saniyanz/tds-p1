import datetime
import os

def handle_A3():
    """
    Task A3:
    - Read dates from /data/dates.txt (one per line).
    - Count the number of dates that fall on a Wednesday.
    - Write the count (as a number) to /data/dates-wednesdays.txt.
    """
    input_file = "/data/dates.txt"
    output_file = "/data/dates-wednesdays.txt"

    # Ensure the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    count_wednesdays = 0

    with open(input_file, "r") as f:
        for line in f:
            date_str = line.strip()
            if not date_str:
                continue  # Skip empty lines

            try:
                # Attempt to parse the date assuming the format is YYYY-MM-DD.
                # Adjust the format if your dates have a different pattern.
                dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                # If parsing fails, you might consider using a more flexible parser.
                # For example, using dateutil (pip install python-dateutil):
                from dateutil.parser import parse
                try:
                    dt = parse(date_str)
                except Exception as parse_error:
                    raise ValueError(f"Failed to parse date '{date_str}': {parse_error}")

            # In Python's datetime, Monday is 0 and Wednesday is 2.
            if dt.weekday() == 2:
                count_wednesdays += 1

    # Write the count to the output file
    with open(output_file, "w") as f:
        f.write(str(count_wednesdays))

    return f"A3 completed: {count_wednesdays} Wednesdays found."
