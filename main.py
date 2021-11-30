import argparse
from skimage import io
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse arguments from command line into ARGS."""

    parser = argparse.ArgumentParser(
        description="The runner for our polygonization program",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data',
        help='Path to input image',
        dest='data',
        default='data' + os.sep + 'boats.jpg'
    )

    return parser.parse_args()


def main() -> None:
    """Interprets arguments and begins polygonization."""
    args = parse_args() 
    
    # Placeholder: read and display args.data
    img = io.imread(args.data)
    io.imshow(img)
    plt.show()

    return
    


if __name__ == "__main__":
    main()
