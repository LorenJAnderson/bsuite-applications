from environment import DeepSea


def load_deep_sea(id):
    if id == 0:
        return DeepSea(n=1)
    else:
        return DeepSea(n=3)