import sys
from pathlib import Path

import torch
from PIL import Image


# -------------------------------------------------------------------
# FastReID source
# -------------------------------------------------------------------

FASTREID_ROOT = Path(__file__).resolve().parents[2] / "reid" / "fast-reid"

if str(FASTREID_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTREID_ROOT))


from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.transforms import build_transforms


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

CONFIG_PATH = (
    FASTREID_ROOT
    / "configs"
    / "Market1501"
    / "bagtricks_R50.yml"
)

WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "reid"
    / "models"
    / "market_bot_R50.pth"
)


class ReIDExtractor:
    """
    FastReID-based person Re-ID feature extractor.

    Input:
        Person image/crop

    Output:
        2048-dimensional visual embedding
    """

    def __init__(self, device="cuda"):

        self.device = device if torch.cuda.is_available() else "cpu"

        # -----------------------------------------------------------
        # Load configuration
        # -----------------------------------------------------------

        self.cfg = get_cfg()
        self.cfg.merge_from_file(str(CONFIG_PATH))

        self.cfg.MODEL.WEIGHTS = str(WEIGHTS_PATH)
        self.cfg.MODEL.DEVICE = self.device

        # -----------------------------------------------------------
        # Build model
        # -----------------------------------------------------------

        self.model = build_model(self.cfg)

        # -----------------------------------------------------------
        # Load trained Re-ID weights
        # -----------------------------------------------------------

        Checkpointer(self.model).load(self.cfg.MODEL.WEIGHTS)

        self.model.eval()

        # -----------------------------------------------------------
        # Build FastReID preprocessing
        # -----------------------------------------------------------

        self.transform = build_transforms(
            self.cfg,
            is_train=False
        )

    def extract(self, image_path):
        """
        Extract a Re-ID embedding from an image.

        Parameters
        ----------
        image_path : str or Path
            Path to a person crop.

        Returns
        -------
        torch.Tensor
            2048-dimensional normalized embedding.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # -----------------------------------------------------------
        # Load image
        # -----------------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # -----------------------------------------------------------
        # FastReID preprocessing
        # -----------------------------------------------------------

        image = self.transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)

        # Move to GPU/CPU
        image = image.to(self.device)

        # -----------------------------------------------------------
        # Extract embedding
        # -----------------------------------------------------------

        with torch.no_grad():
            features = self.model(image)

        # -----------------------------------------------------------
        # Normalize embedding
        # -----------------------------------------------------------

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=1
        )

        return features[0].cpu()


if __name__ == "__main__":
    print("Re-ID extractor module loaded successfully.")