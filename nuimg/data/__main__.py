from argparse import ArgumentParser, BooleanOptionalAction

from nuimg.data.id_export import export as idexp
from nuimg.data.ood_export import export as oodexp

PARSER = ArgumentParser()
PARSER.add_argument("--ood", action=BooleanOptionalAction)


if __name__ == "__main__":
    args = PARSER.parse_args()

    if args.ood:
        oodexp()
    else:
        idexp()
