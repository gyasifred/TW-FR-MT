"""Local application entry point; network sharing is opt-in."""

from twfr_mt.cli import main

if __name__ == "__main__":
    main(["serve"])
