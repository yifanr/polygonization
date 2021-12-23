import argparse
import os
import time

import matplotlib.pyplot as plt

from polygonize import polygonize


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

    parser.add_argument(
        '--evaluate',
        help='Evaluate average runtime of polygonization over 10 runs',
        dest='evaluate',
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '--clusters',
        help='Number of clusters for k-means clustering',
        dest='clusters',
        type=int,
        default=10
    )

    parser.add_argument(
        '--percent',
        help='Percent of points with highest edge score to select vertices from',
        dest='percent',
        type=float,
        default=5
    )

    parser.add_argument(
        '--vertices',
        help='Number of vertices for final triangulation',
        dest='vertices',
        type=int,
        default=2500
    )

    return parser.parse_args()


def main() -> None:
    """Interprets arguments and begins polygonization."""
    # Parse arguments
    args = parse_args()

    # Polygonize
    if (args.evaluate):
        # Evaluate mode
        start = time.time()
        for i in range(10):
            plt.clf()
            res = polygonize(args.data, args.clusters, args.vertices)
        print("Average time per run: " + str((time.time() - start) / 10))
    else:
        # Normal execution
        res = polygonize(args.data, args.clusters, args.vertices, args.percent)

    plt.imshow(res)
    plt.show()

    return


if __name__ == "__main__":
    main()
