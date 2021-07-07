import torch


def save_model(save_name, epoch, loss, acc, model, optimizer):
    state = {
        "epoch": epoch,
        "loss": loss,
        "acc": acc,
        "weight": model.state_dict(),
        "optimizer": optimizer.state_dict()["param_groups"],
    }
    torch.save(state, save_name)
