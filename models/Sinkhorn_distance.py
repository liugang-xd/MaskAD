import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)


class SinkhornDistance(nn.Module):


    def __init__(self, eps, max_iter, dis, device, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.dis = dis
        self.device = device

    def forward(self, x, y):
        if self.dis == 'cos':
            C = self._cost_matrix(x, y, 'cos')
        elif self.dis == 'euc':
            C = self._cost_matrix(x, y, 'euc')
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).to(self.device).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).to(self.device).squeeze()
        u = torch.zeros_like(mu).to(self.device)
        v = torch.zeros_like(nu).to(self.device)
        actual_nits = 0
        thresh = 1e-1
        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        dis = (pi * C).sum(dim=1)
        cost = torch.sum(pi * C, dim=(-2, -1))
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return dis, cost, pi, C

    def M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    def _cost_matrix(self, x, y, dis, p=2):
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if dis == 'cos':
            C = 1 - d_cosine(x_col, y_lin)
        elif dis == 'euc':
            C = torch.mean((torch.abs(x_col - y_lin)) ** p, -1)

        return C
