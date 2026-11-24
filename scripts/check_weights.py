from hashlib import file_digest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "vgg_normalised.pth":
        "804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a",
    "experiment/final_exp/decoder_final.pth":
        "11a5f1811a503388f9ee246610e1b573089493dad7fa1efd4838d603c9150378",
    "experiment/pilot_600/decoder_step_000200.pth":
        "020fd531a263a3d53b965d213f3888147056009f377376d81d33690be6c52b70",
}


def main():
    valid = True
    for name, expected in EXPECTED.items():
        try:
            with (ROOT / name).open("rb") as weight:
                matches = file_digest(weight, "sha256").hexdigest() == expected
            status = "OK" if matches else "CHECKSUM MISMATCH"
        except OSError as error:
            status = str(error)
        print(f"{name}: {status}")
        valid = valid and status == "OK"
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
