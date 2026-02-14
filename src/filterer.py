
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FiltererConfig:
    method: str = "default"

class Filterer:
    def __init__(self, cfg: FiltererConfig = FiltererConfig()):
        self.cfg = cfg

    def process_folder(self, in_dir, out_dir):
        pass


def main():
    print("Filtering images to ensure image quality...")
    # TODO: implement filters

if __name__ == "__main__":
    main()
