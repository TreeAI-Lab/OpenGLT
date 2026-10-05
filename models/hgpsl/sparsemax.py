import torch
import torch.nn as nn
from torch.autograd import Function
from torch_scatter import scatter_add, scatter_max


def scatter_sort(x, batch, fill_value=-1e16):
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max().item())
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes,), fill_value)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    sorted_x, _ = dense_x.sort(dim=-1, descending=True)
    cumsum_sorted_x = sorted_x.cumsum(dim=-1).view(-1)
    sorted_x = sorted_x.view(-1)
    filled_index = sorted_x != fill_value
    return sorted_x[filled_index], cumsum_sorted_x[filled_index]


def _make_ix_like(batch):
    num_nodes = scatter_add(batch.new_ones(batch.size(0)), batch, dim=0)
    return torch.cat([
        torch.arange(1, int(count.item()) + 1, dtype=torch.long, device=batch.device)
        for count in num_nodes
    ], dim=0)


def _threshold_and_support(x, batch):
    """Return the sparsemax threshold for each (possibly interleaved) group.

    HGPSL passes source-node ids as ``batch``.  They are not guaranteed to be
    contiguous in edge order.  The original dense scatter-sort assumed that
    equal ids were already adjacent, which silently overwrote slots and could
    produce a shorter sorted vector than ``x``.  Sort by value first and then
    stably by group so every group is contiguous without changing its internal
    descending order.
    """
    if x.dim() != 1 or batch.dim() != 1 or x.numel() != batch.numel():
        raise ValueError(
            "Grouped sparsemax expects equally-sized 1-D score and group tensors; "
            f"got x={tuple(x.shape)}, batch={tuple(batch.shape)}."
        )
    if x.numel() == 0:
        return x.new_zeros(0), batch.new_zeros(0)

    # Stable two-key sort: descending score inside ascending group id.
    value_order = torch.argsort(x, descending=True, stable=True)
    group_order = torch.argsort(batch[value_order], stable=True)
    order = value_order[group_order]
    sorted_input = x[order]
    sorted_batch = batch[order]

    dim_size = int(batch.max().item()) + 1
    counts = scatter_add(
        batch.new_ones(batch.numel()), batch, dim=0, dim_size=dim_size
    )
    starts = counts.cumsum(0) - counts
    positions = (
        torch.arange(x.numel(), device=x.device, dtype=batch.dtype)
        - starts[sorted_batch]
        + 1
    )

    global_cumsum = sorted_input.cumsum(0)
    prefix = torch.cat([global_cumsum.new_zeros(1), global_cumsum], dim=0)
    grouped_cumsum = global_cumsum - prefix[starts][sorted_batch]
    support = positions.to(x.dtype) * sorted_input > (grouped_cumsum - 1.0)
    support_size = scatter_add(
        support.to(batch.dtype), sorted_batch, dim=0, dim_size=dim_size
    )

    last_support = starts + support_size.clamp(min=1) - 1
    tau = x.new_zeros(dim_size)
    nonempty = counts > 0
    tau[nonempty] = (
        grouped_cumsum[last_support[nonempty]] - 1.0
    ) / support_size[nonempty].to(x.dtype)
    return tau, support_size


class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, x, batch):
        max_val, _ = scatter_max(x, batch)
        x = x - max_val[batch]
        tau, supp_size = _threshold_and_support(x, batch)
        output = torch.clamp(x - tau[batch], min=0)
        ctx.save_for_backward(supp_size, output, batch)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output, batch = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = scatter_add(grad_input, batch) / supp_size.clamp(min=1).to(output.dtype)
        grad_input = torch.where(output != 0, grad_input - v_hat[batch], grad_input)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):
    def forward(self, x, batch):
        return sparsemax(x, batch)
