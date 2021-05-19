import torch
import torch.nn as nn


class TFC(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, in_channels, num_layers, gr, kt, kf, activation):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        activation: activation function
        """
        super(TFC, self).__init__()

        c = in_channels
        self.H = nn.ModuleList()
        for i in range(num_layers):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=gr, kernel_size=(kf, kt), stride=1,
                              padding=(kt // 2, kf // 2)),
                    nn.BatchNorm2d(gr),
                    activation(),
                )
            )
            c += gr

        self.activation = self.H[-1][-1]

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, gr, T, F] """
        x_ = self.H[0](x)
        for h in self.H[1:]:
            x = torch.cat((x_, x), 1)
            x_ = h(x)

        return x_


class DTFC(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, in_channels, out_channels, num_layers, gr, kt, kf, activation):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        activation: activation function
        """
        super(DTFC, self).__init__()

        assert num_layers > 2
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=gr, kernel_size=(kf, kt), stride=1,
                      padding=(kt // 2, kf // 2)),
            nn.BatchNorm2d(gr),
            activation(),
        )

        c = gr
        d = 1
        self.H = nn.ModuleList()
        for i in range(num_layers - 2):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=gr, kernel_size=(kf, kt), stride=1,
                              padding=((kt // 2) * d, (kf // 2) * d), dilation=d),
                    nn.BatchNorm2d(gr),
                    activation(),
                )
            )
            c += gr
            d += 2

        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=(kf, kt), stride=1,
                      padding=(kt // 2, kf // 2)),
            nn.BatchNorm2d(out_channels),
            activation(),
        )

        self.activation = self.H[-1][-1]

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, gr, T, F] """
        x = self.first_conv(x)

        for h in self.H:
            x_ = h(x)
            x = torch.cat((x_, x), 1)

        return self.last_conv(x)


class TDF(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, channels, f, bn_factor=16, bias=False, min_bn_units=16, activation=nn.ReLU):

        """
        channels: # channels
        f: num of frequency bins
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers
        activation: activation function
        """

        super(TDF, self).__init__()
        if bn_factor is None:
            self.tdf = nn.Sequential(
                nn.Linear(f, f, bias),
                nn.BatchNorm2d(channels),
                activation()
            )

        else:
            bn_units = max(f // bn_factor, min_bn_units)
            self.bn_units = bn_units
            self.tdf = nn.Sequential(
                nn.Linear(f, bn_units, bias),
                nn.BatchNorm2d(channels),
                activation(),
                nn.Linear(bn_units, f, bias),
                nn.BatchNorm2d(channels),
                activation()
            )

    def forward(self, x):
        return self.tdf(x)


class TFC_TDF(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor=16, min_bn_units=16, bias=False,
                 activation=nn.ReLU):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        f: num of frequency bins

        below are params for TDF
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers

        activation: activation function
        """

        super(TFC_TDF, self).__init__()
        self.tfc = TFC(in_channels, num_layers, gr, kt, kf, activation)
        self.tdf = TDF(gr, f, bn_factor, bias, min_bn_units, activation)
        self.activation = self.tdf.tdf[-1]

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tdf(x)


class DTFC_TDF(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, in_channels, out_channels, num_layers, gr, kt, kf,
                 f, bn_factor=16, min_bn_units=16, bias=False, activation=nn.ReLU):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        f: num of frequency bins

        below are params for TDF
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers

        activation: activation function
        """
        super(DTFC_TDF, self).__init__()
        self.tfc = DTFC(in_channels, out_channels, num_layers, gr, kt, kf, activation)
        self.tdf = TDF(out_channels, f, bn_factor, bias, min_bn_units, activation)
        self.activation = self.tdf.tdf[-1]

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tdf(x)


class TDF_f1_to_f2(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, channels, f1, f2, bn_factor=16, bias=False, min_bn_units=16, activation=nn.ReLU):

        """
        channels:  # channels
        f1: num of frequency bins (input)
        f2: num of frequency bins (output)
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers
        activation: activation function
        """

        super(TDF_f1_to_f2, self).__init__()

        self.num_target_f = f2

        if bn_factor is None:
            self.tdf = nn.Sequential(
                nn.Linear(f1, f2, bias),
                nn.BatchNorm2d(channels),
                activation()
            )

        else:
            bn_units = max(f2 // bn_factor, min_bn_units)
            self.tdf = nn.Sequential(
                nn.Linear(f1, bn_units, bias),
                nn.BatchNorm2d(channels),
                activation(),
                nn.Linear(bn_units, f2, bias),
                nn.BatchNorm2d(channels),
                activation()
            )


    def forward(self, x):
        return self.tdf(x)


class TFC_RNN(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, in_channels, num_layers_tfc, gr, kt, kf,
                 f, bn_factor_rnn, num_layers_rnn, bidirectional=True, min_bn_units_rnn=16, bias_rnn=True,
                 bn_factor_tdf=16, bias_tdf=True,
                 skip_connection=True,
                 activation=nn.ReLU):
        """
        in_channels: number of input channels
        num_layers_tfc: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis

        f: num of frequency bins
        bn_factor_rnn: bottleneck factor of rnn
        num_layers_rnn: number of layers of rnn
        bidirectional: if true then bidirectional version rnn
        bn_factor_tdf: bottleneck factor of tdf
        bias: bias
        skip_connection: if true then tic+rnn else rnn

        activation: activation function
        """

        super(TFC_RNN, self).__init__()

        self.skip_connection = skip_connection

        self.tfc = TFC(in_channels, num_layers_tfc, gr, kt, kf, activation)
        self.bn = nn.BatchNorm2d(gr)

        hidden_units_rnn = max(f // bn_factor_rnn, min_bn_units_rnn)
        self.rnn = nn.GRU(f, hidden_units_rnn, num_layers_rnn, bias=bias_rnn, batch_first=True,
                          bidirectional=bidirectional)

        f_from = hidden_units_rnn * 2 if bidirectional else hidden_units_rnn
        f_to = f
        self.tdf_f1_to_f2 = TDF_f1_to_f2(gr, f_from, f_to, bn_factor=bn_factor_tdf, bias=bias_tdf,
                                         activation=activation)

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, gr, T, F] """

        x = self.tfc(x)  # [B, in_channels, T, F] => [B, gr, T, F]
        x = self.bn(x)  # [B, gr, T, F] => [B, gr, T, F]
        tfc_output = x

        B, C, T, F = x.shape
        x = x.view(-1, T, F)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # [B * gr, T, F] => [B * gr, T, 2*hidden_size]
        x = x.view(B, C, T, -1)  # [B * gr, T, 2*hidden_size] => [B, gr, T, 2*hidden_size]
        rnn_output = self.tdf_f1_to_f2(x)  # [B, gr, T, 2*hidden_size] => [B, gr, T, F]

        return tfc_output + rnn_output if self.skip_connection else rnn_output
