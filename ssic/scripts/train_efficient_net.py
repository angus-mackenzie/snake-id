
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.callbacks.hooks import num_features_model, Learner
from efficientnet_pytorch import EfficientNet
from ranger import Ranger
from ssic.ssic import SSIC


# ========================================================================= #
# ARGS                                                                      #
# ========================================================================= #


class args:
    seed = 42
    lr = 1e-3
    validate_ratio = 0.2
    label_smoothing_eps = 0.1
    lookahead_steps = 6
    batch_size = 32
    epochs = 40

    model_name = 'efficientnet-b0'  # b0-b7
    # activation = 'relu'             # TODO: relu / mish
    # optimizer = 'ranger'            # TODO: ranger / radam / adam / rmsprop


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def make_model(out_features):
    model = EfficientNet.from_pretrained(args.model_name)
    # TODO: should 1280 not use num_features_model?
    model.add_module('_fc', nn.Linear(1280, out_features))
    return model

def make_data():
    imagelist = SSIC.get_train_imagelist(args.validate_ratio)
    imagelist = imagelist.transform(
        # TODO: I saw some function somehwere that added proven fastai defaults
        tfms=([rotate(degrees=(-90, 90), p=1)],[]),
        size=EfficientNet.get_image_size(args.model_name),
        resize_method=ResizeMethod.SQUISH
    )
    data = imagelist.databunch(bs=args.batch_size).normalize(imagenet_stats)
    # TODO: data.show_batch(3, figsize=(9, 9))
    return data

def make_learner():
    data = make_data()
    model = make_model(out_features=data.c)
    # noinspection PyArgumentList
    learner = Learner(
        data=data,
        model=model,
        loss_func=LabelSmoothingCrossEntropy(args.label_smoothing_eps),
        opt_func=partial(Ranger, lr=args.lr, k=args.lookahead_steps),
        metrics=[accuracy, FBeta(beta=1, average='macro')]
    )
    # TODO: I assume this is for multiple GPUs?
    # learn.split([
    #     [learn.model._conv_stem, learn.model._bn0, learn.model._blocks[:8]],
    #     [learn.model._blocks[8:], learn.model._conv_head],
    #     [learn.model._bn1, learn.model._fc]
    # ])
    return learner


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    SSIC.init(seed=42)
    learner = make_learner()
    fit_one_cycle(
        learn=learner,
        cyc_len=args.epochs,
        max_lr=slice(args.lr/100, args.lr)
    )
    learner.save('b0-trained-for-40', return_path=True)
    learner = learner.load('b0-trained-for-40')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
