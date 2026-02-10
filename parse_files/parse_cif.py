import re
import sys
import json
import os


#this script parse .cif file to get atom positions, symmetry operations
#.cif file is generated from material studio

paramErrCode = 3         # Wrong command-line parameters
fileNotExistErrCode = 4  # Configuration file doesn't exist
if len(sys.argv) != 2:
    print("wrong number of arguments.", file=sys.stderr)
    print("usage: python parse_cif.py /path/to/xxx.cif", file=sys.stderr)
    exit(paramErrCode)

cif_file_name = str(sys.argv[1])
# Check if configuration file exists
if not os.path.exists(cif_file_name):
    print(f"file not found: {cif_file_name}", file=sys.stderr)
    exit(fileNotExistErrCode)


def remove_comments_and_empty_lines_cif(file):
    """
    Remove comments and empty lines from a CIF-like configuration file.

    - Comments start with # and continue to end of line.
    - # inside single (') or double (") quotes are preserved.
    - Empty lines (or lines with only whitespace) are removed.

    :param file: conf file path
    :return: list of cleaned lines
    """
    with open(file, "r") as fptr:
        lines = fptr.readlines()

    linesToReturn = []
    # Regex explanation:
    # 1. (["\'])  -> Capture group 1: Match a quote (single or double)
    # 2. (?:      -> Non-capturing group for content inside quotes
    #    \\.      -> Match escaped characters (like \")
    #    |        -> OR
    #    [^\\]    -> Match any character that isn't a backslash
    #    )*?      -> Match as few times as possible (lazy)
    # 3. \1       -> Match the closing quote (same as group 1)
    # 4. |        -> OR
    # 5. (#.*$)   -> Capture group 2: Match a real comment (starts with #, goes to end)
    # This pattern finds quoted strings OR comments.
    # We will use a callback to decide what to do.
    pattern = re.compile(r'(["\'])(?:\\.|[^\\])*?\1|(#.*$)')
    for oneLine in lines:
        # We use a callback function for re.sub
        # If group 2 (the comment) exists, replace it with empty string.
        # If group 1 (the quote) matches, return the whole match (preserve it).
        def replace_func(match):
            if match.group(2):
                return ""  # It's a comment, remove it
            else:
                return match.group(0)  # It's a quoted string, keep it
        # Apply the regex
        cleaned_line = pattern.sub(replace_func, oneLine).strip()
        # Only add non-empty lines
        if cleaned_line:
            linesToReturn.append(cleaned_line)

    return linesToReturn

coef_pattern = re.compile(r"([+-]?)\s*([xyz])", re.IGNORECASE)

# Matches: optional sign, optional space, number (integer, decimal, or fraction)
# Example: "+1/2", "-0.5", "1/4", "+1"
translation_pattern = re.compile(r"([+-]?)\s*(\d+(?:[./]\d+)?)")

# The Master Regex
# Group 1,2: Variable (sign, char)
# Group 3,4: Number (sign, val)
term_pattern = re.compile(r"([+-]?)\s*([xyz])|([+-]?)\s*(\d+(?:[./]\d+)?)", re.IGNORECASE)

def parse_single_expression(expression_str):
    """
    Parses a string like '-x+1/2' or 'x-y' into numerical coefficients.
    :param expression_str:
    :return: (coeff_x, coeff_y, coeff_z, translation)
    """
    # Initialize values
    c_x = 0.0
    c_y = 0.0
    c_z = 0.0
    trans = 0.0
    # Find all matches in the string
    for match in term_pattern.finditer(expression_str):
        # --- CASE 1: It's a Variable: x, y, z ---
        if match.group(2):
            sign_str = match.group(1)
            variable = match.group(2).lower()
            # Determine value: +1 or -1
            value = -1.0 if sign_str == '-' else 1.0
            if variable == 'x':
                c_x += value
            elif variable == 'y':
                c_y += value
            elif variable == 'z':
                c_z += value
        # --- CASE 2: It's a Number: Translation ---
        elif match.group(4):
            sign_str = match.group(3)
            number_str = match.group(4)
            # Handle fraction conversion (e.g., "1/2" -> 0.5)
            if '/' in number_str:
                num, den = number_str.split('/')
                val = float(num) / float(den)
            else:
                val = float(number_str)
            # Apply sign
            if sign_str == '-':
                val = -val
            trans += val

    return c_x, c_y, c_z, trans



# # --- TEST EXAMPLES ---
# examples = [
#     "-x+1/2",       # Standard
#     "1/2-x",        # Non-standard (Translation first)
#     "0.5+y",        # Non-standard (Decimal + Variable)
#     "z",            # Simple
#     "-y-x+1/4"      # Complex
# ]
#
# print(f"{'Expression':<15} | {'x':<4} {'y':<4} {'z':<4} {'Trans':<5}",file=sys.stdout)
# print("-" * 45,file=sys.stdout)
#
# for ex in examples:
#     cx, cy, cz, tr = parse_single_expression(ex)
#     print(f"{ex:<15} | {cx:<4} {cy:<4} {cz:<4} {tr:<5}",file=sys.stdout)


#main parsing function
def parse_cif_contents(file):

    # Get cleaned lines from file
    linesWithCommentsRemoved = remove_comments_and_empty_lines_cif(file)
    for line in linesWithCommentsRemoved:
        print(line)




# print(parse_cif_contents(cif_file_name),file=sys.stdout)