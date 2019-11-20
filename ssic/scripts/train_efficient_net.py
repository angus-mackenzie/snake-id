import fastai
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.callbacks.hooks import num_features_model, Learner
from efficientnet_pytorch import EfficientNet

from cnn_layer_visualization import CNNLayerVisualization
from deep_dream import DeepDream
from generate_class_specific_samples import ClassSpecificImageGeneration
from gradcam import GradCam
from guided_backprop import GuidedBackprop

from ranger import Ranger
from radam import RAdam
from Mish.Torch.functional import mish
from Mish.Torch.mish import Mish

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

    load = 'b0-trained-for-40'
    save = False # 'b0-trained-for-40'

    train = True
    visualise = False
    use_gpu = True

    num_classes = 45


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def make_model():
    model = EfficientNet.from_pretrained(args.model_name)
    model.add_module('_fc', nn.Linear(1280, args.num_classes))
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

def make_learner(model, data):
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

    # default device
    defaults.device = torch.device('cuda' if args.use_gpu else 'cpu')

    # model = make_model()
    # model.load_state_dict(torch.load(f'../../notebooks/vanilla_b0_efficientnet/{args.load}.pth', map_location=defaults.device)['model'])



    target_class = 130  # Flamingo
    # pretrained_model = models.alexnet(pretrained=True).to(defaults.device)





    # doesnt really produce good results
    # pretrained_model = EfficientNet.from_pretrained(args.model_name)
    # csig = ClassSpecificImageGeneration(pretrained_model, target_class)
    # csig.generate()

    # EFFICIENT NET DOES NOT SUPPORT LAYER INDEXING
    # cnn_layer = 17
    # filter_pos = 5
    # layer_vis = CNNLayerVisualization(model, cnn_layer, filter_pos)
    # layer_vis.visualise_layer_with_hooks()




    #
    #
    # # CREATE DATA
    # if args.train:
    #     data = make_data()
    #     assert data.c == args.num_classes, 'number of classes mismatch'
    # else:
    #     data = None
    #
    # # MODEL
    # model = make_model(out_features=args.num_classes)
    #
    # create_cnn()
    #
    # learner = make_learner(model, data)
    # if args.load:
    #     learner = learner.load(args.load)
    #
    # # TRAIN
    # if args.train:
    #     fit_one_cycle(
    #         learn=learner,
    #         cyc_len=args.epochs,
    #         max_lr=slice(args.lr/100, args.lr)
    #     )
    #     # SAVE
    #     if args.save:
    #         learner.save(args.save, return_path=True)

    # if args.visualise:
    #     # https://github.com/utkuozbulak/pytorch-cnn-visualizations
    #     # subset of the best methods.
    #     image_gen    = ClassSpecificImageGeneration(model)
    #     layer_vis    = CNNLayerVisualization(model)
    #     guided_prop  = GuidedBackprop(model)
    #     grad_cap     = GradCam(model)
    #     deep_dream   = DeepDream(model)
    #     # TODO: USE!

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
