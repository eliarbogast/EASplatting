import torch

from easplatting.cameras import load_image
from easplatting.render import _rasterize


def learned_gates(
    scene,
    views,
    baselines,
    steps=300,
    batch_size=8,
    learning_rate=0.05,
    start=2.0,
):
    # a keep gate per gaussian, learned by gradient descent: the gated render must
    # match the unpruned render and every open gate costs something
    # returns the gate logits, higher means the gaussian matters more
    device = scene["means"].device
    count = scene["means"].shape[0]
    width, height = views.size
    viewmats, Ks = views.viewmats.to(device), views.Ks.to(device)
    fixed = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in scene.items()}

    logits = torch.full((count,), start, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    sparsity = None
    for step in range(steps):
        # the batch: a few random views
        pick = torch.randperm(len(views), device=device)[:batch_size]
        gated = dict(fixed)
        gated["opacities"] = fixed["opacities"] * torch.sigmoid(logits)
        image, _, _ = _rasterize(gated, viewmats[pick], Ks[pick], width, height)
        error = ((image[..., :3] - baselines[pick]) ** 2).mean()

        # the price of a gate matches the typical reconstruction gradient
        if sparsity is None:
            gradient = torch.autograd.grad(error, logits, retain_graph=True)[0]
            sparsity = gradient.abs()[gradient != 0].median()
        loss = error + sparsity * torch.sigmoid(logits).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return logits.detach()


def fisher_scores(scene, views, photos):
    # sensitivity of the photo loss to each gaussian's opacity, scale and position,
    # summed over views (a diagonal, simplified version of the PUP-3DGS score)
    device = scene["means"].device
    count = scene["means"].shape[0]
    width, height = views.size
    viewmats, Ks = views.viewmats.to(device), views.Ks.to(device)

    score = torch.zeros(count, device=device)
    for i in range(len(views)):
        splats = {k: v.detach() if torch.is_tensor(v) else v for k, v in scene.items()}
        names = ("means", "scales", "opacities")
        for name in names:
            splats[name] = splats[name].clone().requires_grad_()
        view = slice(i, i + 1)
        image, _, _ = _rasterize(splats, viewmats[view], Ks[view], width, height)
        error = ((image[0, ..., :3] - photos[i]) ** 2).mean()
        error.backward()

        for name in names:
            gradient = splats[name].grad
            if gradient is not None:
                score += (gradient**2).reshape(count, -1).sum(dim=1)
    return score


def load_photos(views, device):
    # the ground truth photos of the views at the resolution they are rendered at
    width, height = views.size
    photos = [load_image(path, size=(width, height)) for path in views.image_paths]
    return torch.stack(photos).to(device)
