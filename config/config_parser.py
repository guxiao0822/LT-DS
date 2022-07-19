import ast
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="args for ltds")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="../config/exp/awa2.yaml",
        type=str,
    )

    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type= ast.literal_eval,
        dest = 'auto_resume',
        required=False,
        default= False,
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args