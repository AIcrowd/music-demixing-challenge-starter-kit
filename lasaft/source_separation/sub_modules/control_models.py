import torch.nn as nn

class dense_control_block(nn.Module):

    def __init__(self, input_dim, num_layer, activation=nn.ReLU, scale=2, scale_type="exp"):
        super(dense_control_block, self).__init__()
        self.input_dim = input_dim
        self.num_layer = num_layer
        self.activation = activation

        linear_list = []
        if scale_type == 'exp':
            dims = [input_dim * (scale ** i) for i in range(num_layer)]
        elif scale_type == 'mul':
            dims = [input_dim + input_dim * (scale * i) for i in range(num_layer)]

        for i, (in_features, out_features) in enumerate(zip(dims[:-1], dims[1:])):
            extra = i != 0
            linear_list.append(nn.Linear(in_features, out_features))
            linear_list.append(activation())

            if extra:
                linear_list.append(nn.Dropout())
                linear_list.append(nn.BatchNorm1d(out_features))

        self.linear = nn.Sequential(*linear_list)
        self.last_dim = dims[-1]

    def forward(self, x_condition):
        return self.linear(x_condition)


class pocm_control_model(nn.Module):
    def __init__(self, dense_control_block, n_blocks, internal_channels, gamma_activation=nn.Identity,
                 beta_activation=nn.Identity, pocm_to='full', pocm_norm=None):
        super(pocm_control_model, self).__init__()

        self.dense_control_block = dense_control_block
        self.n_blocks = n_blocks
        self.c = internal_channels
        self.gamma_activation = gamma_activation()
        self.beta_activation = beta_activation()

        if pocm_to == 'full':
            self.full, self.encoder_only, self.decoder_only = True, False, False
            num_target_block = n_blocks
        elif pocm_to == 'encoder':
            self.full, self.encoder_only, self.decoder_only = False, True, False
            num_target_block = n_blocks // 2
        elif pocm_to == 'decoder':
            self.full, self.encoder_only, self.decoder_only = False, False, True
            num_target_block = n_blocks // 2
        else:
            raise NotImplementedError

        assert pocm_norm in [None, 'batch_norm', 'full_batch_norm', 'lstm_like']

        def mk_norm(pocm_norm, type):
            if pocm_norm is None:
                return nn.Identity()
            elif 'batch_norm' in pocm_norm:
                if type == 'gamma':
                    return nn.BatchNorm1d(num_target_block * (internal_channels ** 2), affine=False)
                elif type == 'beta':
                    return nn.BatchNorm1d(num_target_block * internal_channels, affine=False)
            elif 'lstm_like' == pocm_norm:
                if type == 'gamma':
                    return nn.BatchNorm1d(num_target_block * (internal_channels ** 2), affine=False)
                elif type == 'beta':
                    return nn.BatchNorm1d(num_target_block * internal_channels, affine=False)

        self.linear_gamma = nn.Sequential(
            nn.Linear(dense_control_block.last_dim, num_target_block * (internal_channels ** 2)),
            self.gamma_activation,
            mk_norm(pocm_norm, 'gamma')
        )
        self.linear_beta = nn.Sequential(
            nn.Linear(dense_control_block.last_dim, num_target_block * internal_channels),
            self.beta_activation,
            mk_norm(pocm_norm, 'beta')
        )

    def forward(self, x):
        x = self.dense_control_block(x)
        m = self.n_blocks // 2

        if self.full:
            gammas = self.gamma_split(self.linear_gamma(x), 0, self.n_blocks)
            betas = self.beta_split(self.linear_beta(x), 0, self.n_blocks)

            g_encoder, g_middle, g_decoder = gammas[:m], gammas[m], gammas[m + 1:]
            gammas = [g_encoder, g_middle, g_decoder]

            b_encoder, b_middle, b_decoder = betas[:m], betas[m], betas[m + 1:]
            betas = [b_encoder, b_middle, b_decoder]

        elif self.encoder_only or self.decoder_only:
            gammas = self.gamma_split(self.linear_gamma(x), 0, m)
            betas = self.beta_split(self.linear_beta(x), 0, m)

            if self.encoder_only:
                gammas = [gammas, None, None]
                betas = [betas, None, None]
            else:
                gammas = [None, None, gammas]
                betas = [None, None, betas]

        else:
            raise NotImplementedError

        return gammas, betas

    def gamma_split(self, tensor, start_idx, end_idx):
        tensor_shape = list(tensor.shape[:-1]) + [self.c, self.c]
        return [tensor[..., layer * self.c * self.c: (layer + 1) * self.c * self.c].view(tensor_shape)
                for layer in range(start_idx, end_idx)]

    def beta_split(self, tensor, start_idx, end_idx):
        return [tensor[..., layer * self.c: (layer + 1) * self.c]
                for layer in range(start_idx, end_idx)]


class film_control_model(nn.Module):
    def __init__(self,
                 dense_control_block, n_blocks, internal_channels,
                 film_type,
                 gamma_activation=nn.Identity,
                 beta_activation=nn.Identity, condition_to='full'):
        super(film_control_model, self).__init__()

        self.dense_control_block = dense_control_block
        self.n_blocks = n_blocks
        self.c = internal_channels
        self.gamma_activation = gamma_activation()
        self.beta_activation = beta_activation()

        if condition_to == 'full':
            self.full, self.encoder_only, self.decoder_only = True, False, False
            num_target_block = n_blocks
        elif condition_to == 'encoder':
            self.full, self.encoder_only, self.decoder_only = False, True, False
            num_target_block = n_blocks // 2
        elif condition_to == 'decoder':
            self.full, self.encoder_only, self.decoder_only = False, False, True
            num_target_block = n_blocks // 2
        else:
            raise NotImplementedError

        num_unit = internal_channels if film_type == 'complex' else 1
        self.linear_gamma = nn.Sequential(
            nn.Linear(dense_control_block.last_dim, num_target_block * num_unit),
            self.gamma_activation,
        )
        self.linear_beta = nn.Sequential(
            nn.Linear(dense_control_block.last_dim, num_target_block * num_unit),
            self.beta_activation,
        )

    def forward(self, x):
        x = self.dense_control_block(x)
        m = self.n_blocks // 2

        if self.full:
            gammas = self.gamma_split(self.linear_gamma(x), 0, self.n_blocks)
            betas = self.beta_split(self.linear_beta(x), 0, self.n_blocks)

            g_encoder, g_middle, g_decoder = gammas[:m], gammas[m], gammas[m + 1:]
            gammas = [g_encoder, g_middle, g_decoder]

            b_encoder, b_middle, b_decoder = betas[:m], betas[m], betas[m + 1:]
            betas = [b_encoder, b_middle, b_decoder]

        elif self.encoder_only or self.decoder_only:
            gammas = self.gamma_split(self.linear_gamma(x), 0, m)
            betas = self.beta_split(self.linear_beta(x), 0, m)

            if self.encoder_only:
                gammas = [gammas, None, None]
                betas = [betas, None, None]
            else:
                gammas = [None, None, gammas]
                betas = [None, None, betas]

        else:
            raise NotImplementedError

        return gammas, betas

    def gamma_split(self, tensor, start_idx, end_idx):
        return [tensor[..., layer * self.c: (layer + 1) * self.c]
                for layer in range(start_idx, end_idx)]

    def beta_split(self, tensor, start_idx, end_idx):
        return [tensor[..., layer * self.c: (layer + 1) * self.c]
                for layer in range(start_idx, end_idx)]
