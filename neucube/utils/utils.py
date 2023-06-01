def print_summary(info_ls):
    """
    Prints a summary of information in a tabular format.

    Args:
        info_ls (list): A list of lists representing the information to be summarized.
                        Each inner list represents a row of information.

    Example:
        info = [
            ['Name', 'Age', 'Location'],
            ['John Doe', '30', 'New York'],
            ['Jane Smith', '25', 'San Francisco']
        ]
        print_summary(info)

    Output:
        Name        Age   Location
        John Doe    30    New York
        Jane Smith  25    San Francisco
    """
    widths = [max(map(len, col)) for col in zip(*info_ls)]
    for row in info_ls:
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))