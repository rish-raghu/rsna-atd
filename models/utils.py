from . import unet
from . import densenet

def get_model(args):
    if args.arch=='unet2d':
        model = unet.UNet2D(args.z_size)
    elif args.arch=='unet3d':
        model = unet.UNet3D()
    elif args.arch.startswith('densenet'):
        depth = int(args.arch.split('densenet')[1])
        model = densenet.generate_model(depth, batch_norm=args.batch_norm)

    return model
