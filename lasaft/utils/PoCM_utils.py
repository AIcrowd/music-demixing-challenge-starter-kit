import torch
import torch.nn.functional as f


def Pocm_naive(x, gammas, betas):
    """
    :param x: an output feature map of a CNN layer [*, ch, T, F]
    :param gamma: [*, ch, ch]
    :param beta: [*, ch]
    :return: gamma * x + beta
    """
    x = x.unsqueeze(-4)
    gammas = gammas.unsqueeze(-1).unsqueeze(-1)

    pocm = [f.conv2d(x_, gamma_, beta_) for x_, gamma_, beta_ in zip(x, gammas, betas)]

    return torch.cat(pocm, dim=0)


def Pocm_Matmul(x, gammas, betas):
    """
    :param x: an output feature map of a CNN layer [*, ch, T, F]
    :param gamma: [*, ch, ch]
    :param beta: [*, ch]
    :return: gamma * x + beta
    """
    x = x.transpose(-1, -3)  # [*, F, T, ch]
    gammas = gammas.unsqueeze(-3)  # [*, 1, ch, ch]

    pocm = torch.matmul(x, gammas) + betas.unsqueeze(-2).unsqueeze(-3)

    return pocm.transpose(-1, -3)
