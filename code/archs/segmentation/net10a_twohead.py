from IIC.code.archs.segmentation.net10a import SegmentationNet10aHead, SegmentationNet10aTrunk, \
  SegmentationNet10a
from IIC.code.archs.cluster.vgg import VGGNet
import GPUtil

__all__ = ["SegmentationNet10aTwoHead"]


class SegmentationNet10aTwoHead(VGGNet):
  def __init__(self, config):
    super(SegmentationNet10aTwoHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = SegmentationNet10aTrunk(config, cfg=SegmentationNet10a.cfg)
    self.head_A = SegmentationNet10aHead(config, output_k=config.output_k_A,
                                         cfg=SegmentationNet10a.cfg)
    self.head_B = SegmentationNet10aHead(config, output_k=config.output_k_B,
                                         cfg=SegmentationNet10a.cfg)

    self._initialize_weights()

  def forward(self, x, head="B"):
    print("First GPU looksie")
    GPUtil.showUtilization()
    x = self.trunk(x)
    print("Post trunk looksie")
    GPUtil.showUtilization()
    print("Trunk x: " + str(x))
    if head == "A":
      print("Pre GPU head A")
      GPUtil.showUtilization()
      x = self.head_A(x)
      print("Post GPU head A")
      GPUtil.showUtilization()
    elif head == "B":
      print("Pre GPU head B")
      GPUtil.showUtilization()
      x = self.head_B(x)
      print("Post GPU head B")
      GPUtil.showUtilization()
    else:
      assert (False)
    
    print("Final x in head forward: " + str(x))
    return x
