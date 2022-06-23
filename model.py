import math
import torch
import torch.nn as nn
from torch import Tensor

""" Activation Functions """


class AbsoluteMaxGating(nn.Module):
    def __init__(self) -> None:
        super(AbsoluteMaxGating, self).__init__()

    @staticmethod
    def forward(
            neuron_out: Tensor, segment_out: Tensor  # N * D_out, D_out * S
    ) -> Tensor:
        # get the dendritic activation
        index = torch.argmax(torch.abs(segment_out), dim=1)  # D_out
        index = index.reshape(len(segment_out), 1)  # D_out * 1
        percent = segment_out.gather(dim=1, index=index)  # D_out * 1
        percent = percent.reshape(len(segment_out))  # D_out
        percent.sigmoid_()  # D_out

        # modifies x by the dendritic activation
        return torch.mul(neuron_out, percent)  # N * D_out


class kWTA(nn.Module):
    def __init__(self, in_features: int, percent_on: int = 0.05) -> None:
        super(kWTA, self).__init__()
        self.in_features = in_features
        self.percent_on = percent_on
        self.k = int(round(in_features * percent_on))

    def forward(self, x: Tensor) -> Tensor:
        indices = torch.topk(x, k=self.k, dim=1, sorted=False)[1]
        off_mask = torch.ones_like(x, dtype=torch.bool)
        off_mask.scatter_(1, indices, 0)  # set value of "indices" as False
        return x.masked_fill(off_mask, 0)  # if value in off_mask==True, set 0

    def extra_repr(self) -> str:
        return 'in_features={}, percent_on={}, k={}'.format(
            self.in_features, self.percent_on, self.k
        )


""" Dendritic Segments """


class LinearDendriticSegments(nn.Module):
    def __init__(
            self, num_neurons: int, dim_context: int, num_segment: int
    ) -> None:
        super(LinearDendriticSegments, self).__init__()
        self.num_neurons = num_neurons  # D_out
        self.dim_context = dim_context  # C
        self.num_segment = num_segment  # S

        self.weight = nn.Parameter(
            torch.empty((self.num_neurons, self.dim_context, self.num_segment))
        )  # D_out * C * S
        self.bias = nn.Parameter(
            torch.empty((self.num_neurons, self.num_segment))
        )  # D_out * S
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, context: Tensor) -> Tensor:  # C
        return torch.einsum("c,dcs->ds", context, self.weight) + self.bias
        # D_out * S

    def extra_repr(self) -> str:
        return 'num_neurons={}, dim_context={}, num_segment={}'.format(
            self.num_neurons, self.dim_context, self.num_segment
        )

    def update(self) -> None:
        old_data = self.weight.data

        self.num_segment += 1  # S += 1
        self.weight = nn.Parameter(
            torch.empty((self.num_neurons, self.dim_context, self.num_segment))
        )  # D_out * C * S
        self.reset_parameters()

        self.weight.data[:, :, :self.num_segment-1] = old_data


""" Point Neurons """


class LinearPointNeurons(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(LinearPointNeurons, self).__init__()
        self.in_features = in_features  # D_in
        self.out_features = out_features  # D_out

        self.weight = nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:  # N * D_in
        return torch.mm(x, self.weight) + self.bias  # N * D_out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class QuadraticPointNeurons(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(QuadraticPointNeurons, self).__init__()
        self.in_features = in_features  # D_in
        self.out_features = out_features  # D_out

        self.weight_r = nn.Parameter(torch.empty((in_features, out_features)))
        self.weight_g = nn.Parameter(torch.empty((in_features, out_features)))
        self.weight_b = nn.Parameter(torch.empty((in_features, out_features)))
        self.bias_r = nn.Parameter(torch.empty(out_features))
        self.bias_g = nn.Parameter(torch.empty(out_features))
        self.bias_b = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight, bias in ((self.weight_r, self.bias_r),
                             (self.weight_g, self.bias_g),
                             (self.weight_b, self.bias_b)):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:  # N * D_in
        z_r = torch.mm(x, self.weight_r) + self.bias_r  # N * D_out
        z_g = torch.mm(x, self.weight_g) + self.bias_g  # N * D_out
        z_b = torch.mm(x ** 2, self.weight_b) + self.bias_b  # N * D_out
        return torch.mul(z_r, z_g) + z_b  # N * D_out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


""" Active Dendrites """


class LinearActiveDendrites(nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            dim_context: int, num_segment: int,
    ) -> None:
        super(LinearActiveDendrites, self).__init__()
        self.in_features = in_features  # D_in
        self.out_features = out_features  # D_out
        self.dim_context = dim_context  # C
        self.num_segment = num_segment  # S

        self.neuron = LinearPointNeurons(in_features, out_features)
        self.segment = LinearDendriticSegments(
            out_features, dim_context, num_segment
        )
        self.gating = AbsoluteMaxGating()

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        neuron_out = self.neuron(x)
        segment_out = self.segment(context)
        return self.gating(neuron_out, segment_out)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, '.format(
            self.in_features, self.out_features
        ) + 'dim_context={}, num_segment={}'.format(
            self.dim_context, self.num_segment
        )

    def update(self):
        self.segment.update()


class QuadraticActiveDendrites(nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            dim_context: int, num_segment: int
    ) -> None:
        super(QuadraticActiveDendrites, self).__init__()
        self.in_features = in_features  # D_in
        self.out_features = out_features  # D_out
        self.dim_context = dim_context  # C
        self.num_segment = num_segment  # S

        self.neuron = QuadraticPointNeurons(in_features, out_features)
        self.segment = LinearDendriticSegments(
            out_features, dim_context, num_segment
        )
        self.gating = AbsoluteMaxGating()

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        neuron_out = self.neuron(x)
        segment_out = self.segment(context)
        return self.gating(neuron_out, segment_out)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, '.format(
            self.in_features, self.out_features
        ) + 'dim_context={}, num_segment={}'.format(
            self.dim_context, self.num_segment
        )

    def update(self):
        self.segment.update()


""" Networks """


class LinearNeuralNet(nn.Module):
    def __init__(self, features: list or tuple) -> None:
        super(LinearNeuralNet, self).__init__()
        self.features = features
        self.L = len(features) - 1  # number of layers (including output)

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(self.L - 1):
            self.layers.append(
                LinearPointNeurons(self.features[i], self.features[i + 1])
            )
            self.activations.append(nn.ReLU())
        self.layers.append(
            LinearPointNeurons(self.features[-2], self.features[-1])
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for i in range(self.L - 1):
            out = self.layers[i](out)
            out = self.activations[i](out)
        return self.layers[-1](out)

    def extra_repr(self) -> str:
        return 'features={}'.format(self.features)


class QuadraticNeuralNet(nn.Module):
    def __init__(self, features: list or tuple) -> None:
        super(QuadraticNeuralNet, self).__init__()
        self.features = features
        self.L = len(features) - 1  # number of layers (including output)

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(self.L - 1):
            self.layers.append(
                QuadraticPointNeurons(self.features[i], self.features[i + 1])
            )
            self.activations.append(nn.ReLU())
        self.layers.append(
            QuadraticPointNeurons(self.features[-2], self.features[-1])
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for i in range(self.L - 1):
            out = self.layers[i](out)
            out = self.activations[i](out)
        return self.layers[-1](out)

    def extra_repr(self) -> str:
        return 'features={}'.format(self.features)


class LinearDendriticNet(nn.Module):
    def __init__(
            self, features: list or tuple, num_segment: int = None,
            prototype: Tensor = None, threshold: int = 0.99,
            percent_on: int = 0.05
    ) -> None:
        super(LinearDendriticNet, self).__init__()

        # network
        self.features: list or tuple = features
        self.L: int = len(features) - 1  # number of layers (including output)

        # prototype
        self.given_prototype: bool = False if prototype is None else True
        self.prototype: Tensor = Tensor() if prototype is None else prototype
        self.num_prototype: int = 0 if prototype is None else len(prototype)
        self.store_x: list = []
        self.threshold: int = threshold

        # dendritic segments
        self.fix_num_segment: bool = True if num_segment is not None else False
        self.num_segment: int
        if num_segment is not None: self.num_segment = num_segment
        elif prototype is not None: self.num_segment = len(prototype)
        else: self.num_segment = 1

        # activation function
        self.percent_on: int = percent_on

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(self.L - 1):
            self.layers.append(
                LinearActiveDendrites(self.features[i], self.features[i + 1],
                                      self.features[0], self.num_segment)
            )
            self.activations.append(kWTA(self.features[i + 1], self.percent_on))
        self.layers.append(
            LinearActiveDendrites(self.features[-2], self.features[-1],
                                  self.features[0], self.num_segment)
        )

    def forward(self, x: Tensor, test: bool = False) -> Tensor:
        update = False
        if test is True or self.given_prototype is True:
            context = self.choose_context(x)
        else:
            context, update = self.cluster(x)
        if self.fix_num_segment is False and update is True: self.update()

        out = x
        for i in range(self.L - 1):
            out = self.layers[i](out, context)
            out = self.activations[i](out)
        return self.layers[-1](out, context)

    def extra_repr(self) -> str:
        return 'features={}, num_prototype={}, threshold={}, '.format(
            self.features, self.num_prototype, self.threshold
        ) + 'fix_num_segment={}, percent_on={}'.format(
            self.fix_num_segment, self.percent_on
        )

    """ Update number of segment in dendritic segments """

    def update(self) -> None:
        if self.num_segment == self.num_prototype: return None
        self.num_segment += 1
        assert self.num_segment == self.num_prototype
        for layer in self.layers: layer.update()

    """ Inferring prototypes during training and get context vector """

    def is_match(self, x: Tensor, y: Tensor) -> bool:  # N_x * D, N_y * D
        x_n, d = x.shape  # N_x, D
        y_n, _ = y.shape  # N_y

        mean_x = torch.mean(x, dim=0, keepdim=True)  # 1 * D
        mean_y = torch.mean(y, dim=0, keepdim=True)  # 1 * D
        mean_diff = mean_x - mean_y  # 1 * D
        centered_x = x - mean_x  # N_x * D
        centered_y = y - mean_y  # N_y * D

        cov_x = torch.t(centered_x) @ centered_x  # D * D
        cov_y = torch.t(centered_y) @ centered_y  # D * D
        pooled_cov = (cov_x + cov_y) / (x_n + y_n - 2)  # D * D

        t2 = mean_diff @ torch.pinverse(pooled_cov) @ torch.t(mean_diff)
        t2 = (x_n * y_n * t2) / (x_n + y_n)
        f = (t2 * (x_n + y_n - d - 1)) / (d * (x_n + y_n - 2))
        return f <= self.threshold

    def cluster(self, x: Tensor) -> (Tensor, bool):
        for task in range(self.num_prototype - 1, -1, -1):
            if self.is_match(x, self.store_x[task]):
                self.store_x[task] = torch.cat((x, self.store_x[task]))
                self.prototype[task] = torch.mean(self.store_x[task], dim=0)
                return self.prototype[task], False
        self.store_x.append(x)
        self.prototype = torch.cat((
            self.prototype,
            torch.mean(self.store_x[self.num_prototype], dim=0, keepdim=True)
        ))
        self.num_prototype += 1
        return self.prototype[-1], True

    """ Get context vector when given prototypes or during testing """

    def choose_context(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, dim=0, keepdim=True)  # 1 * D
        index = torch.argmax(torch.norm((mean-self.prototype), dim=1, p=2))
        return self.prototype[index]


class QuadraticDendriticNet(nn.Module):
    def __init__(
            self, features: list or tuple, num_segment: int = None,
            prototype: Tensor = None, threshold: int = 0.99,
            percent_on: int = 0.05
    ) -> None:
        super(QuadraticDendriticNet, self).__init__()

        # network
        self.features: list or tuple = features
        self.L: int = len(features) - 1  # number of layers (including output)

        # prototype
        self.given_prototype: bool = False if prototype is None else True
        self.prototype: Tensor = Tensor() if prototype is None else prototype
        self.num_prototype: int = 0 if prototype is None else len(prototype)
        self.store_x: list = []
        self.threshold: int = threshold

        # dendritic segments
        self.fix_num_segment: bool = True if num_segment is not None else False
        self.num_segment: int
        if num_segment is not None: self.num_segment = num_segment
        elif prototype is not None: self.num_segment = len(prototype)
        else: self.num_segment = 1

        # activation function
        self.percent_on: int = percent_on

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(self.L - 1):
            self.layers.append(
                QuadraticActiveDendrites(self.features[i], self.features[i + 1],
                                         self.features[0], self.num_segment)
            )
            self.activations.append(kWTA(self.features[i + 1], self.percent_on))
        self.layers.append(
            QuadraticActiveDendrites(self.features[-2], self.features[-1],
                                     self.features[0], self.num_segment)
        )

    def forward(self, x: Tensor, test: bool = False) -> Tensor:
        update = False
        if test is True or self.given_prototype is True:
            context = self.choose_context(x)
        else:
            context, update = self.cluster(x)
        if self.fix_num_segment is False and update is True: self.update()

        out = x
        for i in range(self.L - 1):
            out = self.layers[i](out, context)
            out = self.activations[i](out)
        return self.layers[-1](out, context)

    def extra_repr(self) -> str:
        return 'features={}, num_prototype={}, threshold={}, '.format(
            self.features, self.num_prototype, self.threshold
        ) + 'fix_num_segment={}, percent_on={}'.format(
            self.fix_num_segment, self.percent_on
        )

    """ Update number of segment in dendritic segments """

    def update(self) -> None:
        if self.num_segment == self.num_prototype: return None
        self.num_segment += 1
        assert self.num_segment == self.num_prototype
        for layer in self.layers: layer.update()

    """ Inferring prototypes during training and get context vector """

    def is_match(self, x: Tensor, y: Tensor) -> bool:  # N_x * D, N_y * D
        x_n, d = x.shape  # N_x, D
        y_n, _ = y.shape  # N_y

        mean_x = torch.mean(x, dim=0, keepdim=True)  # 1 * D
        mean_y = torch.mean(y, dim=0, keepdim=True)  # 1 * D
        mean_diff = mean_x - mean_y  # 1 * D
        centered_x = x - mean_x  # N_x * D
        centered_y = y - mean_y  # N_y * D

        cov_x = torch.t(centered_x) @ centered_x  # D * D
        cov_y = torch.t(centered_y) @ centered_y  # D * D
        pooled_cov = (cov_x + cov_y) / (x_n + y_n - 2)  # D * D

        t2 = mean_diff @ torch.pinverse(pooled_cov) @ torch.t(mean_diff)
        t2 = (x_n * y_n * t2) / (x_n + y_n)
        f = (t2 * (x_n + y_n - d - 1)) / (d * (x_n + y_n - 2))
        return f <= self.threshold

    def cluster(self, x: Tensor) -> (Tensor, bool):
        for task in range(self.num_prototype - 1, -1, -1):
            if self.is_match(x, self.store_x[task]):
                self.store_x[task] = torch.cat((x, self.store_x[task]))
                self.prototype[task] = torch.mean(self.store_x[task], dim=0)
                return self.prototype[task], False
        self.store_x.append(x)
        self.prototype = torch.cat((
            self.prototype,
            torch.mean(self.store_x[self.num_prototype], dim=0, keepdim=True)
        ))
        self.num_prototype += 1
        return self.prototype[-1], True

    """ Get context vector when given prototypes or during testing """

    def choose_context(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, dim=0, keepdim=True)  # 1 * D
        index = torch.argmax(torch.norm((mean-self.prototype), dim=1, p=2))
        return self.prototype[index]
