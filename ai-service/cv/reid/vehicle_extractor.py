import sys
from pathlib import Path

import torch
from PIL import Image


FASTREID_ROOT = (
    Path(__file__).resolve().parents[2]
    / "reid"
    / "fast-reid"
)

if str(FASTREID_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTREID_ROOT))


from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.transforms import build_transforms


CONFIG_PATH = (
    FASTREID_ROOT
    / "configs"
    / "VeRi"
    / "sbs_R50-ibn.yml"
)

WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "reid"
    / "models"
    / "veri_sbs_R50-ibn.pth"
)


class VehicleReIDExtractor:

    def __init__(self, device="cuda"):

        self.device = (
            device
            if torch.cuda.is_available()
            else "cpu"
        )

        self.cfg = get_cfg()

        self.cfg.merge_from_file(
            str(CONFIG_PATH)
        )

        self.cfg.MODEL.WEIGHTS = str(
            WEIGHTS_PATH
        )

        self.cfg.MODEL.DEVICE = self.device

        self.model = build_model(
            self.cfg
        )

        Checkpointer(self.model).load(
            self.cfg.MODEL.WEIGHTS
        )

        self.model.eval()

        self.transform = build_transforms(
            self.cfg,
            is_train=False
        )

    def extract(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(
            image
        )

        image = image.unsqueeze(0)

        image = image.to(
            self.device
        )

        with torch.no_grad():

            features = self.model(
                image
            )

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=1
        )

        return features[0].cpu()

    def extract_from_array(self, image):

        """
        Extract a vehicle Re-ID embedding
        directly from an OpenCV BGR image.
        """

        image = Image.fromarray(
            image[:, :, ::-1]
        ).convert("RGB")

        image = self.transform(
            image
        )

        image = image.unsqueeze(0)

        image = image.to(
            self.device
        )

        with torch.no_grad():

            features = self.model(
                image
            )

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=1
        )

        return features[0].cpu()


if __name__ == "__main__":

    print(
        "Vehicle Re-ID extractor "
        "module loaded successfully."
    )