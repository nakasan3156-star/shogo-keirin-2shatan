"""章悟式∞競輪OS 個人評価型3PDF API。"""

from .keirin_individual_api import VERSION, predict
from .keirin_pdf_adapter import predict_from_files

__all__ = ["VERSION", "predict", "predict_from_files"]
