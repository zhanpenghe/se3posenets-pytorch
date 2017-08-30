import configargparse

def setup_comon_options():
    # Parse arguments
    parser = configargparse.ArgumentParser(description='SE3-Pose-Nets Training')

    # Dataset options
    parser.add_argument('-c', '--config', required=True, is_config_file=True,
                        help='Path to config file for parameters')
    parser.add_argument('-d', '--data', default=[], required=True,
                        action='append', metavar='DIRS', help='path to dataset(s), passed in as list [a,b,c...]')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-j', '--num-workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train-per', default=0.6, type=float,
                        metavar='FRAC', help='fraction of data for the training set (default: 0.6)')
    parser.add_argument('--val-per', default=0.15, type=float,
                        metavar='FRAC', help='fraction of data for the validation set (default: 0.15)')
    parser.add_argument('--img-scale', default=1e-4, type=float,
                        metavar='IS', help='conversion scalar from depth resolution to meters (default: 1e-4)')
    parser.add_argument('--step-len', default=1, type=int,
                        metavar='N', help='number of frames separating each example in the training sequence (default: 1)')
    parser.add_argument('--seq-len', default=1, type=int,
                        metavar='N', help='length of the training sequence (default: 1)')
    parser.add_argument('--ctrl-type', default='actdiffvel', type=str,
                        metavar='STR', help='Control type: actvel | actacc | comvel | comacc | comboth | [actdiffvel] | comdiffvel')

    # Model options
    parser.add_argument('--no-batch-norm', action='store_true', default=False,
                        help='disables batch normalization (default: False)')
    parser.add_argument('--pre-conv', action='store_true', default=False,
                        help='puts batch normalization and non-linearity before the convolution / de-convolution (default: False)')
    parser.add_argument('--nonlin', default='prelu', type=str,
                        metavar='NONLIN', help='type of non-linearity to use: [prelu] | relu | tanh | sigmoid | elu')
    parser.add_argument('--se3-type', default='se3aa', type=str,
                        metavar='SE3', help='SE3 parameterization: [se3aa] | se3quat | se3spquat | se3euler | affine')
    parser.add_argument('--pred-pivot', action='store_true', default=False,
                        help='Predict pivot in addition to the SE3 parameters (default: False)')
    parser.add_argument('-n', '--num-se3', type=int, default=8,
                        help='Number of SE3s to predict (default: 8)')
    parser.add_argument('--init-transse3-iden', action='store_true', default=False,
                        help='Initialize the weights for the SE3 prediction layer of the transition model to predict identity')
    parser.add_argument('--init-posese3-iden', action='store_true', default=False,
                        help='Initialize the weights for the SE3 prediction layer of the pose-mask model to predict identity')
    parser.add_argument('--local-delta-se3', action='store_true', default=False,
                        help='Predicted delta-SE3 operates in local co-ordinates not global co-ordinates, '
                             'so if we predict "D", full-delta = P1 * D * P1^-1, P2 = P1 * D')
    parser.add_argument('--use-ntfm-delta', action='store_true', default=False,
                        help='Uses the variant of the NTFM3D layer that computes the weighted avg. delta')
    parser.add_argument('--wide-model', action='store_true', default=False,
                        help='Wider network')
    parser.add_argument('--decomp-model', action='store_true', default=False,
                        help='Use a separate encoder for predicting the pose and masks')

    # Mask options
    parser.add_argument('--use-wt-sharpening', action='store_true', default=False,
                        help='use weight sharpening for the mask prediction (instead of the soft-mask model) (default: False)')
    parser.add_argument('--sharpen-start-iter', default=0, type=int,
                        metavar='N', help='Start the weight sharpening from this training iteration (default: 0)')
    parser.add_argument('--sharpen-rate', default=1.0, type=float,
                        metavar='W', help='Slope of the weight sharpening (default: 1.0)')
    parser.add_argument('--use-sigmoid-mask', action='store_true', default=False,
                        help='treat each mask channel independently using the sigmoid non-linearity. Pixel can belong to multiple masks (default: False)')

    # Loss options
    parser.add_argument('--loss-type', default='mse', type=str,
                        metavar='STR', help='Type of loss to use for 3D point errors, only works if we are not using '
                                            'soft-masks (default: mse | abs, normmsesqrt, normmsesqrtpt )')
    parser.add_argument('--motion-norm-loss', action='store_true', default=False,
                        help='normalize the losses by number of points that actually move instead of size average (default: False)')
    parser.add_argument('--consis-wt', default=0.1, type=float,
                        metavar='WT', help='Weight for the pose consistency loss (default: 0.1)')
    parser.add_argument('--loss-scale', default=10000, type=float,
                        metavar='WT', help='Default scale factor for all the losses (default: 1000)')

    # Training options
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--use-pin-memory', action='store_true', default=False,
                        help='Use pin memory - note that this uses additional CPU RAM (default: False)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--train-ipe', default=2000, type=int, metavar='N',
                        help='number of training iterations per epoch (default: 2000)')
    parser.add_argument('--val-ipe', default=500, type=int, metavar='N',
                        help='number of validation iterations per epoch (default: 500)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Optimization options
    parser.add_argument('-o', '--optimization', default='adam', type=str,
                        metavar='OPTIM', help='type of optimization: sgd | [adam]')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-decay', default=0.1, type=float, metavar='M',
                        help='Decay learning rate by this value every decay-epochs (default: 0.1)')
    parser.add_argument('--decay-epochs', default=30, type=int,
                        metavar='M', help='Decay learning rate every this many epochs (default: 10)')

    # Display/Save options
    parser.add_argument('--disp-freq', '-p', default=25, type=int,
                        metavar='N', help='print/disp/save frequency (default: 25)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-s', '--save-dir', default='results', type=str,
                        metavar='PATH', help='directory to save results in. If it doesnt exist, will be created. (default: results/)')

    # Return
    return parser