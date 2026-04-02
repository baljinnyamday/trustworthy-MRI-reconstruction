"""Train MRI reconstruction models at 4x and 8x acceleration."""

from src.mri.config import TrainConfig
from src.mri.train import train


def main() -> None:
    # Train 4x model
    config_4x = TrainConfig(acceleration=4)
    train(config_4x)

    # Train 8x model
    config_8x = TrainConfig(acceleration=8)
    train(config_8x)


if __name__ == "__main__":
    main()
