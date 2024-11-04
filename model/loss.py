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
        one = one.reshape(-1, one.size(2))  # (batch_size * dim_series, len_series)
        another = another.reshape(-1, another.size(2))  # (batch_size * dim_series, len_reduce)
        one_reduce = one_reduce.reshape(-1, one_reduce.size(2)) # (batch_size * dim_series, len_series)
        another_reduce = another_reduce.reshape(-1, another_reduce.size(2))  # (batch_size * dim_series, len_reduce)
        
        original_l2 = self.l2(one, another) / self.scale_factor_original    # (batch_size * dim_series)
        reduce_l2 = self.l2(one_reduce, another_reduce) / self.scale_factor_reduce      # (batch_size * dim_series)

        return self.l1(original_l2.reshape(1, -1), reduce_l2.reshape(1, -1))[0] / one.shape[0]      # (1, batch_size * dim_series) -> scalar