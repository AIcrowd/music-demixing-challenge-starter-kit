def FiLM_simple(x, gamma, beta):
    """
    :param x: an output feature map of a CNN layer [*, ch, T, F]
    :param gamma: [*]
    :param beta: [*]
    :return: gamma * x + beta
    """
    gamma_ = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    beta_ = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return gamma_ * x + beta_


def FiLM_complex(x, gammas, betas):
    """
    :param x: an output feature map of a CNN layer [*, ch, T, F]
    :param gamma: [*, ch]
    :param beta: [*, ch]
    :return: gamma * x + beta
    """
    gamma_ = gammas.unsqueeze(-1).unsqueeze(-1)
    beta_ = betas.unsqueeze(-1).unsqueeze(-1)
    return gamma_ * x + beta_
