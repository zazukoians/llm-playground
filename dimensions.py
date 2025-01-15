#!/usr/bin/env python3
import argparse
from collections import defaultdict

from rdflib import Graph


def format_cube_dimensions(input_file: str) -> str:
    """
    Transform TTL file into a formatted text showing cubes and their dimensions.
    Uses full URLs for cubes and plain text for dimensions.
    """
    # Initialize RDF graph
    g = Graph()
    g.parse(input_file, format="turtle")

    # Group dimensions by cube
    cube_dimensions = defaultdict(set)
    for s, _, o in g:
        subject = str(s)
        obj = str(o)

        # Clean up object value (remove language tags and quotes)
        if obj.endswith('"@en'):
            obj = obj[1:-4]  # Remove quotes and @en

        # Use full URL as cube identifier
        cube_dimensions[subject].add(obj)

    # Format output
    output = []
    for cube, dimensions in sorted(cube_dimensions.items()):
        output.append(f"<{cube}> has dimensions:")
        for dim in sorted(dimensions):
            output.append(f"{dim}")
        output.append("")  # Empty line between cubes

    return "\n".join(output).rstrip()  # rstrip to remove trailing newline

def main():
    parser = argparse.ArgumentParser(description='Format TTL cube dimensions')
    parser.add_argument('input_file', help='Input TTL file path')
    parser.add_argument('output_file', help='Output text file path')
    args = parser.parse_args()

    try:
        # Format dimensions
        formatted_text = format_cube_dimensions(args.input_file)

        # Write output
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_text)

        print(f"Successfully formatted dimensions from {args.input_file} to {args.output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
