from numpy import sqrt
from torch import nn, squeeze
from torch.nn.modules.distance import PairwiseDistance

class ScaledL2Loss(nn.Module):
    def __init__(self, len_series: int, len_reduce: int):
        super(ScaledL2Loss, self).__init__()

        self.l2 = PairwiseDistance(p=2).cuda()
        self.l1 = PairwiseDistance(p=1).cuda()
        self.scale_factor_original = sqrt(len_series)
        self.scale_factor_reduce = sqrt(len_reduce)


    def forward(self, one, another, one_reduce, another_reduce):
        original_l2 = self.l2(squeeze(one), squeeze(another)) / self.scale_factor_original
        reduce_l2 = self.l2(squeeze(one_reduce), squeeze(another_reduce)) / self.scale_factor_reduce
        return self.l1(original_l2.view([1, -1]), reduce_l2.view([1, -1]))[0] / database.shape[0]