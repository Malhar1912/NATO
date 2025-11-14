def adjust_learning_rate(optimizer, epoch: int, lr_decay_epoch: int = 10) -> None:
    """
    Adjusts learning rates in optimizer's param groups in-place.

    Notes:
      - Each group should have 'initial_lr' present.
    """
    for group in optimizer.param_groups:
        initial_lr = group.get("initial_lr", group.get("lr", 1e-3))
        decay_step = group.get("decay_step", lr_decay_epoch)
        lr_decay = group.get("lr_decay", 0.1)
        if decay_step > 0:
            group['lr'] = initial_lr * (lr_decay ** (epoch // decay_step))
        else:
            group['lr'] = initial_lr
