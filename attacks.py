import torch
import torch.nn.functional as F

def FLAG(model_forward, perturb_shape, y, gamma, opt, args):
    model, forward = model_forward
    model.train()
    opt.zero_grad()
    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size)
    perturb.requires_grad_()
    yl, zl, yd, zd = forward(perturb)
    losspl = F.binary_cross_entropy(yl, y)
    losspd = F.binary_cross_entropy(yd, y.t())
    loss = gamma * losspl + (1 - gamma) * losspd
    for _ in range(args.m):
        loss.backward(retain_graph=True)
        perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.detach())
        perturb.data = perturb_data.data
        model.train()
        yz, zl, yd, zd = forward(perturb)
        losspl = F.binary_cross_entropy(yl, y)
        losspd = F.binary_cross_entropy(yd, y.t())
        loss = gamma * losspl + (1 - gamma) * losspd
    opt.zero_grad()
    loss.backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        yl, _, yd, _ = forward(perturb)
    return loss, yl, yd


