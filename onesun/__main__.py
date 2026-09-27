"""Command line: ``python -m onesun {download,run,retrieve} ...``."""

import argparse

from onesun import pgn
from onesun.experiment import load_config, retrieve_record, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m onesun", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="build or extend an L0 archive")
    download.add_argument("location", help="PGN location name, e.g. Rome-SAP")
    download.add_argument("pandora", type=int, help="Pandora number, e.g. 117")
    download.add_argument("start", help="first day, YYYY-MM-DD")
    download.add_argument("end", help="last day (inclusive), YYYY-MM-DD")
    download.add_argument("--spectrometer", type=int, default=1)
    download.add_argument("--data-dir", default="data")

    run = commands.add_parser("run", help="fit, retrieve and evaluate from a config")
    run.add_argument("config", help="experiment JSON, see examples/")
    run.add_argument("--no-download", action="store_true",
                     help="use the existing archive without contacting the PGN API")

    apply = commands.add_parser("retrieve", help="apply a saved model to a date range")
    apply.add_argument("model", help="model.npz written by 'run'")
    apply.add_argument("key", help="instrument key used in the experiment config")
    apply.add_argument("location")
    apply.add_argument("pandora", type=int)
    apply.add_argument("start")
    apply.add_argument("end")
    apply.add_argument("--spectrometer", type=int, default=1)
    apply.add_argument("--data-dir", default="data")
    apply.add_argument("--output", default=None)

    args = parser.parse_args()
    if args.command == "download":
        instrument = pgn.Instrument(args.location, args.pandora, args.spectrometer)
        pgn.build_l0_archive(instrument, args.start, args.end, args.data_dir)
    elif args.command == "run":
        output = run_experiment(load_config(args.config), download=not args.no_download)
        print(f"results in {output}")
    else:
        path = retrieve_record(args.model, args.key, args.location, args.pandora,
                               args.spectrometer, args.start, args.end,
                               args.data_dir, args.output)
        print(f"retrievals written to {path}")


if __name__ == "__main__":
    main()
