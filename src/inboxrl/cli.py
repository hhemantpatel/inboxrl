"""`inboxrl` console entry point: `inboxrl extract ...` and `inboxrl eval ...`."""

from __future__ import annotations

import sys


def main() -> None:
    commands = {
        "extract": "inboxrl.demo.extract",
        "eval": "inboxrl.evaluate",
        "train": "inboxrl.train_grpo",
        "synth": "inboxrl.data.synth",
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"usage: inboxrl {{{','.join(commands)}}} [args...]   (use `inboxrl <cmd> --help` for details)")
        sys.exit(1)
    import importlib

    cmd = sys.argv.pop(1)
    sys.argv[0] = f"inboxrl {cmd}"
    importlib.import_module(commands[cmd]).main()


if __name__ == "__main__":
    main()
