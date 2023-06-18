from ..builder import DETECTORS
from .cascade_rcnn import CascadeRCNN


@DETECTORS.register_module()
class ConvNeXt(CascadeRCNN):
    """Implementation of `ConvNeXt <https://arxiv.org/pdf/2201.03545.pdf>`_"""

    def __init__(self, **kwargs):
        super(ConvNeXt, self).__init__(**kwargs)
