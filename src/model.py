# Auto-extracted model code for NeRF (positional encoding + model classes)
import torch


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input."""
    encoding = [tensor] if include_input else []
    for i in range(num_encoding_functions):
        for func in [torch.sin, torch.cos]:
            encoding.append(func(2. ** i * tensor))
    return torch.cat(encoding, dim=-1)


class VeryTinyNeRFModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers."""

    def __init__(
        self, filter_size=128, num_encoding_fn_xyz=6, num_encoding_fn_dir=4,
        use_viewdirs=True, include_input_xyz=True, include_input_dir=True
    ):
        super(VeryTinyNeRFModel, self).__init__()

        self.include_input_xyz = 3 if include_input_xyz else 0
        self.include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = self.include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = self.include_input_dir + 2 * 3 * num_encoding_fn_dir
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_dir, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that (almost) follows the figure from the paper (smaller)."""

    def __init__(
        self, hidden_size=128, num_encoding_fn_xyz=6, num_encoding_fn_dir=4,
        include_input_xyz=True, include_input_dir=True
    ):
        super(ReplicateNeRFModel, self).__init__()

        self.include_input_xyz = 3 if include_input_xyz else 0
        self.include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = self.include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = self.include_input_dir + 2 * 3 * num_encoding_fn_dir

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, 1 + hidden_size)
        self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.layer6 = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz = x[..., :self.dim_xyz]
        direction = x[..., self.dim_xyz:]
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        sigma_feat = self.layer3(x_)
        sigma = sigma_feat[..., 0].unsqueeze(-1)
        feat = sigma_feat[..., 1:]
        y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        y_ = self.relu(self.layer5(y_))
        y_ = self.layer6(y_)
        return torch.cat((y_, sigma), dim=-1)