from torch import optim


def add_weight_decay(model, lr_custm=1e-4, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    decay_lr = []
    no_decay_lr = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "prd" in name or "mrtd" in name:
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay_lr.append(param)
            else:
                decay_lr.append(param)
        else:
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay_lr, "weight_decay": 0.0, "lr": lr_custm},
        {"params": decay_lr, "weight_decay": weight_decay, "lr": lr_custm},
    ]


def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else {}
        parameters = add_weight_decay(model, args.lr_custm, weight_decay, skip)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    opt_args = {"lr": args.lr, "weight_decay": weight_decay}
    if hasattr(args, "opt_eps") and args.opt_eps is not None:
        opt_args["eps"] = args.opt_eps
    if hasattr(args, "opt_betas") and args.opt_betas is not None:
        opt_args["betas"] = args.opt_betas
    if hasattr(args, "opt_args") and args.opt_args is not None:
        opt_args.update(args.opt_args)

    if opt_lower in {"sgd", "nesterov"}:
        opt_args.pop("eps", None)
        return optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    if opt_lower == "momentum":
        opt_args.pop("eps", None)
        return optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    if opt_lower == "adam":
        return optim.Adam(parameters, **opt_args)
    if opt_lower == "adamw":
        return optim.AdamW(parameters, **opt_args)
    if opt_lower == "adadelta":
        return optim.Adadelta(parameters, **opt_args)
    if opt_lower == "rmsprop":
        return optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)

    raise ValueError(f"Unsupported optimizer: {args.opt}")
