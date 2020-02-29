# Global imports
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import random

# Torch imports
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import torchvision
torch.multiprocessing.set_sharing_strategy('file_system')

# Local imports
import data
import ctrlnets
import flownets
import se3nets
import util
from util import AverageMeter, Tee, DataEnumerator
import helperfuncs as helpers

# InfoGain imports
from data_se3 import DataSE3
from torch.utils.data import DataLoader

#### Setup options
# Common
import argparse
import options
parser = options.setup_comon_options()

# Loss options
parser.add_argument('--init-flow-iden', action='store_true', default=False,
                    help='Initialize the flow network to predict zero flow at the start (default: False)')
parser.add_argument('--pt-wt', default=1, type=float,
                    metavar='WT', help='Weight for the 3D point loss - only FWD direction (default: 1)')
parser.add_argument('--use-se3-nets', action='store_true', default=False,
                    help='Use SE3 nets instead of flow nets (default: False)')

# Define xrange
try:
    a = xrange(1)
except NameError: # Not defined in Python 3.x
    def xrange(*args):
        return iter(range(*args))

################ MAIN
#@profile
def main():
    # Parse args
    global args, num_train_iter
    args = parser.parse_args()
    args.cuda       = not args.no_cuda and torch.cuda.is_available()
    args.batch_norm = not args.no_batch_norm
    assert(args.seq_len == 1)

    ### Create save directory and start tensorboard logger
    util.create_dir(args.save_dir)  # Create directory
    now = time.strftime("%c")
    tblogger = util.TBLogger(args.save_dir + '/logs/' + now)  # Start tensorboard logger

    # Create logfile to save prints
    logfile = open(args.save_dir + '/logs/' + now + '/logfile.txt', 'w')
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, logfile)

    ########################
    ############ Parse options
    # Set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Get default options & camera intrinsics
    args.cam_intrinsics, args.cam_extrinsics, args.ctrl_ids = [], [], []
    args.state_labels = []
    # for k in xrange(len(args.data)):
    #     load_dir = args.data[k] #args.data.split(',,')[0]
    #     try:
    #         # Read from file
    #         intrinsics = data.read_intrinsics_file(load_dir + "/intrinsics.txt")
    #         print("Reading camera intrinsics from: " + load_dir + "/intrinsics.txt")
    #         args.img_ht, args.img_wd = 240, 320  # All data is at 240x320 resolution
    #         args.img_scale = 1.0 / intrinsics['s']  # Scale of the image (use directly from the data)

    #         # Setup camera intrinsics
    #         sc = float(args.img_ht) / intrinsics['ht']  # Scale factor for the intrinsics
    #         cam_intrinsics = {'fx': intrinsics['fx'] * sc,
    #                           'fy': intrinsics['fy'] * sc,
    #                           'cx': intrinsics['cx'] * sc,
    #                           'cy': intrinsics['cy'] * sc}
    #         print("Scale factor for the intrinsics: {}".format(sc))
    #     except:
    #         print("Could not read intrinsics file, reverting to default settings")
    #         args.img_ht, args.img_wd, args.img_scale = 240, 320, 1e-4
    #         cam_intrinsics = {'fx': 589.3664541825391 / 2,
    #                           'fy': 589.3664541825391 / 2,
    #                           'cx': 320.5 / 2,
    #                           'cy': 240.5 / 2}
    #     print("Intrinsics => ht: {}, wd: {}, fx: {}, fy: {}, cx: {}, cy: {}".format(args.img_ht, args.img_wd,
    #                                                                                 cam_intrinsics['fx'],
    #                                                                                 cam_intrinsics['fy'],
    #                                                                                 cam_intrinsics['cx'],
    #                                                                                 cam_intrinsics['cy']))

    #     # Compute intrinsic grid
    #     cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd,
    #                                                                           cam_intrinsics)

    #     # Compute intrinsics
    #     cam_extrinsics = data.read_cameradata_file(load_dir + '/cameradata.txt')

    #     # Get dimensions of ctrl & state
    #     try:
    #         statelabels, ctrllabels, trackerlabels = data.read_statectrllabels_file(load_dir + "/statectrllabels.txt")
    #         print("Reading state/ctrl joint labels from: " + load_dir + "/statectrllabels.txt")
    #     except:
    #         statelabels = data.read_statelabels_file(load_dir + '/statelabels.txt')['frames']
    #         ctrllabels = statelabels  # Just use the labels
    #         trackerlabels = []
    #         print("Could not read statectrllabels file. Reverting to labels in statelabels file")
    #     #args.num_state, args.num_ctrl, args.num_tracker = len(statelabels), len(ctrllabels), len(trackerlabels)
    #     #print('Num state: {}, Num ctrl: {}'.format(args.num_state, args.num_ctrl))
    #     args.num_ctrl = len(ctrllabels)
    #     print('Num ctrl: {}'.format(args.num_ctrl))

    #     # Find the IDs of the controlled joints in the state vector
    #     # We need this if we have state dimension > ctrl dimension and
    #     # if we need to choose the vals in the state vector for the control
    #     ctrlids_in_state = torch.LongTensor([statelabels.index(x) for x in ctrllabels])
    #     print("ID of controlled joints in the state vector: ", ctrlids_in_state.view(1, -1))

    #     # Add to list of intrinsics
    #     args.cam_intrinsics.append(cam_intrinsics)
    #     args.cam_extrinsics.append(cam_extrinsics)
    #     args.ctrl_ids.append(ctrlids_in_state)
    #     args.state_labels.append(statelabels)


    # camera intrinsics:
    datas, loaders = {}, {}
    data_path = '../shapenet5-final/'
    for split in ['train', 'test']:
        datas[split] = DataSE3(data_path=data_path, split=split, seq_len=10, direction_num=10, object_num=6, select_num=None, use_color=False)
        loaders[split] = DataLoader(
            dataset=datas[split],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )

    infogain_camera_intrinsics = datas['train'].camera_intr
    print(infogain_camera_intrinsics)
    args.img_ht, args.img_wd = 240, 320
    # convert to se3 intrinsic
    cam_intrinsics = {
        'fx': infogain_camera_intrinsics[0, 0],
        'fy': infogain_camera_intrinsics[1, 1],
        'cx': infogain_camera_intrinsics[0, 2],
        'cy': infogain_camera_intrinsics[1, 2],
    }
    cam_intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(args.img_ht, args.img_wd, cam_intrinsics)
    camera_extrinsics = datas['train'].camera_pose

    # Data noise
    if not hasattr(args, "add_noise_data") or (len(args.add_noise_data) == 0):
        args.add_noise_data = [False for k in xrange(len(args.data))] # By default, no noise
    else:
        assert(len(args.data) == len(args.add_noise_data))
    if hasattr(args, "add_noise") and args.add_noise: # BWDs compatibility
        args.add_noise_data = [True for k in xrange(len(args.data))]

    # Get mean/std deviations of dt for the data
    if args.mean_dt == 0:
        args.mean_dt = args.step_len * (1.0 / 30.0)
        args.std_dt = 0.005  # +- 10 ms
        print("Using default mean & std.deviation based on the step length. Mean DT: {}, Std DT: {}".format(
            args.mean_dt, args.std_dt))
    else:
        exp_mean_dt = (args.step_len * (1.0 / 30.0))
        assert ((args.mean_dt - exp_mean_dt) < 1.0 / 30.0), \
            "Passed in mean dt ({}) is very different from the expected value ({})".format(
                args.mean_dt, exp_mean_dt)  # Make sure that the numbers are reasonable
        print("Using passed in mean & std.deviation values. Mean DT: {}, Std DT: {}".format(
            args.mean_dt, args.std_dt))

    # Image suffix
    args.img_suffix = '' if (args.img_suffix == 'None') else args.img_suffix # Workaround since we can't specify empty string in the yaml
    print('Ht: {}, Wd: {}, Suffix: {}, Num ctrl: {}'.format(args.img_ht, args.img_wd, args.img_suffix, args.num_ctrl))

    # Read mesh ids and camera data
    # args.baxter_labels = data.read_statelabels_file(args.data[0] + '/statelabels.txt')
    # args.mesh_ids      = args.baxter_labels['meshIds']

    # SE3 stuff
    assert (args.se3_type in ['se3euler', 'se3aa', 'se3quat', 'affine', 'se3spquat']), 'Unknown SE3 type: ' + args.se3_type
    args.se3_dim = ctrlnets.get_se3_dimension(args.se3_type, args.pred_pivot)
    print('Predicting {} SE3s of type: {}. Dim: {}'.format(args.num_se3, args.se3_type, args.se3_dim))

    # Sequence stuff
    print('Step length: {}, Seq length: {}'.format(args.step_len, args.seq_len))

    # Loss parameters
    print('Loss scale: {}, Loss weights => PT: {}, CONSIS: {}'.format(
        args.loss_scale, args.pt_wt, args.consis_wt))

    # Weight sharpening stuff
    if args.use_wt_sharpening:
        print('Using weight sharpening to encourage binary mask prediction. Start iter: {}, Rate: {}'.format(
            args.sharpen_start_iter, args.sharpen_rate))

    # Loss type
    delta_loss = ', Penalizing the delta-flow loss per unroll'
    norm_motion = ', Normalizing loss based on GT motion' if args.motion_norm_loss else ''
    print('3D loss type: ' + args.loss_type + norm_motion + delta_loss)

    # Wide model
    if args.wide_model:
        print('Using a wider network!')

    if args.use_jt_angles:
        print("Using Jt angles as input to the pose encoder")

    if args.use_jt_angles_trans:
        print("Using Jt angles as input to the transition model")

    # DA threshold / winsize
    print("Flow/visibility computation. DA threshold: {}, DA winsize: {}".format(args.da_threshold,
                                                                                 args.da_winsize))
    if args.use_only_da_for_flows:
        print("Computing flows using only data-associations. Flows can only be computed for visible points")
    else:
        print("Computing flows using tracker poses. Can get flows for all input points")

    ########################
    ############ Load datasets
    # Get datasets
    load_color = None
    if args.reject_left_motion:
        print("Examples where any joint of the left arm moves by > 0.005 radians inter-frame will be discarded. \n"
              "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
    if args.reject_right_still:
        print("Examples where no joint of the right arm move by > 0.015 radians inter-frame will be discarded. \n"
              "NOTE: This test will be slow on any machine where the data needs to be fetched remotely")
    if args.add_noise:
        print("Adding noise to the depths, actual configs & ctrls")
    # noise_func = lambda d, c: data.add_gaussian_noise(d, c, std_d=0.02,
    #                                                  scale_d=True, std_j=0.02) if args.add_noise else None
    noise_func = lambda d: data.add_edge_based_noise(d, zthresh=0.04, edgeprob=0.35,
                                                     defprob=0.005, noisestd=0.005)
    valid_filter = lambda p, n, st, se, slab: data.valid_data_filter(p, n, st, se, slab,
                                                               mean_dt=args.mean_dt, std_dt=args.std_dt,
                                                               reject_left_motion=args.reject_left_motion,
                                                               reject_right_still=args.reject_right_still)
    # baxter_data     = data.read_recurrent_baxter_dataset(args.data, args.img_suffix,
    #                                                      step_len = args.step_len, seq_len = args.seq_len,
    #                                                      train_per = args.train_per, val_per = args.val_per,
    #                                                      valid_filter = valid_filter,
    #                                                      cam_extrinsics=args.cam_extrinsics,
    #                                                      cam_intrinsics=args.cam_intrinsics,
    #                                                      ctrl_ids=args.ctrl_ids,
    #                                                      state_labels=args.state_labels,
    #                                                      add_noise=args.add_noise_data)
    # disk_read_func  = lambda d, i: data.read_baxter_sequence_from_disk(d, i, img_ht = args.img_ht, img_wd = args.img_wd,
    #                                                                    img_scale = args.img_scale, ctrl_type = args.ctrl_type,
    #                                                                    num_ctrl=args.num_ctrl,
    #                                                                    #num_state=args.num_state,
    #                                                                    mesh_ids = args.mesh_ids,
    #                                                                    #ctrl_ids=ctrlids_in_state,
    #                                                                    #camera_extrinsics = args.cam_extrinsics,
    #                                                                    #camera_intrinsics = args.cam_intrinsics,
    #                                                                    compute_bwdflows=False,
    #                                                                    load_color=load_color,
    #                                                                    #num_tracker=args.num_tracker,
    #                                                                    dathreshold=args.da_threshold, dawinsize=args.da_winsize,
    #                                                                    use_only_da=args.use_only_da_for_flows,
    #                                                                    noise_func=noise_func) # Need BWD flows / masks if using GT masks

    # train_dataset = data.BaxterSeqDataset(baxter_data, disk_read_func, 'train')  # Train dataset
    # val_dataset   = data.BaxterSeqDataset(baxter_data, disk_read_func, 'val')  # Val dataset
    # test_dataset  = data.BaxterSeqDataset(baxter_data, disk_read_func, 'test')  # Test dataset
    print('==> dataset loaded')
    print('[size] = {0} + {1}'.format(len(datas['train']), len(datas['test'])))

    # Create a data-collater for combining the samples of the data into batches along with some post-processing
    # if args.evaluate:
    #     # Load only test loader
    #     args.imgdisp_freq = 10 * args.disp_freq  # Tensorboard log frequency for the image data
    #     sampler = torch.utils.data.dataloader.SequentialSampler(test_dataset)  # Run sequentially along the test dataset
    #     # torch.manual_seed(args.seed)
    #     # if args.cuda:
    #     #     torch.cuda.manual_seed(args.seed)
    #     # sampler = torch.utils.data.dataloader.RandomSampler(test_dataset) # Random sampler
    #     test_loader = DataEnumerator(util.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
    #                                                  num_workers=args.num_workers, sampler=sampler,
    #                                                  pin_memory=args.use_pin_memory,
    #                                                  collate_fn=test_dataset.collate_batch))
    # else:
    #     # Create dataloaders (automatically transfer data to CUDA if args.cuda is set to true)
    #     train_loader = DataEnumerator(util.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                                   num_workers=args.num_workers, pin_memory=args.use_pin_memory,
    #                                                   collate_fn=train_dataset.collate_batch))
    #     val_loader = DataEnumerator(util.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
    #                                                 num_workers=args.num_workers, pin_memory=args.use_pin_memory,
    #                                                 collate_fn=val_dataset.collate_batch))

    train_loader, val_loader, test_loader = loaders['train'], loaders['test'], loaders['test']
    ########################
    ############ Load models & optimization stuff

    print('Using state of controllable joints')
    args.num_state_net = args.num_ctrl # Use only the jt angles of the controllable joints

    ## Num input channels
    num_input_channels = 3 # Num input channels

    ### Load the model
    num_train_iter = 0
    if args.use_se3_nets:
        model = se3nets.SE3Model(num_ctrl=12, num_se3=args.num_se3,
                        se3_type=args.se3_type, use_pivot=args.pred_pivot,
                        input_channels=num_input_channels, use_bn=args.batch_norm, nonlinearity=args.nonlin,
                        init_transse3_iden=args.init_transse3_iden,
                        use_wt_sharpening=args.use_wt_sharpening, sharpen_start_iter=args.sharpen_start_iter,
                        sharpen_rate=args.sharpen_rate, pre_conv=args.pre_conv,
                        wide=args.wide_model, use_jt_angles=args.use_jt_angles,
                        num_state=args.num_state_net)
    else:
        model = flownets.FlowNet(num_ctrl=args.num_ctrl, num_state=args.num_state_net,
                                 input_channels=num_input_channels, use_bn=args.batch_norm, pre_conv=args.pre_conv,
                                 nonlinearity=args.nonlin, init_flow_iden=args.init_flow_iden,
                                 use_jt_angles=args.use_jt_angles)
    if args.cuda:
        model.cuda() # Convert to CUDA if enabled

    ### Load optimizer
    optimizer = helpers.load_optimizer(args.optimization, model.parameters(), lr=args.lr,
                               momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        # TODO: Save path to TB log dir, save new log there again
        # TODO: Reuse options in args (see what all to use and what not)
        # TODO: Use same num train iters as the saved checkpoint
        # TODO: Print some stats on the training so far, reset best validation loss, best epoch etc
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint       = torch.load(args.resume)
            loadargs         = checkpoint['args']
            args.start_epoch = checkpoint['epoch']
            num_train_iter   = checkpoint['train_iter']
            try:
                model.load_state_dict(checkpoint['state_dict']) # BWDs compatibility (TODO: remove)
            except:
                model.load_state_dict(checkpoint['model_state_dict'])
            assert (loadargs.optimization == args.optimization), "Optimizer in saved checkpoint ({}) does not match current argument ({})".format(
                    loadargs.optimization, args.optimization)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}, train iter {})"
                  .format(args.resume, checkpoint['epoch'], num_train_iter))
            best_epoch = checkpoint['best_epoch'] if hasattr(checkpoint, 'best_epoch') else 0
            print('==== Best validation loss: {} was from epoch: {} ===='.format(checkpoint['best_loss'],
                                                                                 best_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ########################
    ############ Test (don't create the data loader unless needed, creates 4 extra threads)
    if args.evaluate:
        # TODO: Move this to before the train/val loader creation??
        print('==== Evaluating pre-trained network on test data ===')
        test_stats = iterate(test_loader, model, tblogger, len(test_loader), mode='test')

        # Save final test error
        save_checkpoint({
            'args': args,
            'test_stats': {'stats': test_stats,
                           'niters': test_loader.niters, 'nruns': test_loader.nruns,
                           'totaliters': test_loader.iteration_count(),
                           'ids': test_stats.data_ids,
                           },
        }, False, savedir=args.save_dir, filename='test_stats.pth.tar')

        # Close log file & return
        logfile.close()
        return

    ########################
    ############ Train / Validate
    best_val_loss, best_epoch = float("inf") if args.resume == '' else checkpoint['best_loss'], \
                                0 if args.resume == '' else checkpoint['best_epoch']
    if args.resume != '' and hasattr(checkpoint, "best_epoch"):
        best_epoch = checkpoint['best_epoch']
    args.imgdisp_freq = 5 * args.disp_freq # Tensorboard log frequency for the image data
    train_ids, val_ids = [], []
    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.decay_epochs)

        # Train for one epoch
        train_stats = iterate(train_loader, model, tblogger, args.train_ipe,
                           mode='train', optimizer=optimizer, epoch=epoch+1)
        # train_ids += train_stats.data_ids

        # Evaluate on validation set
        val_stats = iterate(val_loader, model, tblogger, args.val_ipe,
                            mode='val', epoch=epoch+1)
        # val_ids += val_stats.data_ids

        # Find best loss
        val_loss = val_stats.loss
        is_best       = (val_loss.avg < best_val_loss)
        prev_best_loss  = best_val_loss
        prev_best_epoch = best_epoch
        if is_best:
            best_val_loss = val_loss.avg
            best_epoch    = epoch+1
            print('==== Epoch: {}, Improved on previous best loss ({}) from epoch {}. Current: {} ===='.format(
                                    epoch+1, prev_best_loss, prev_best_epoch, val_loss.avg))
        else:
            print('==== Epoch: {}, Did not improve on best loss ({}) from epoch {}. Current: {} ===='.format(
                epoch + 1, prev_best_loss, prev_best_epoch, val_loss.avg))

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch+1,
            'args' : args,
            'best_loss'  : best_val_loss,
            'best_epoch' : best_epoch,
            'train_stats': {'stats': train_stats,
                            'niters': train_loader.niters, 'nruns': train_loader.nruns,
                            'totaliters': train_loader.iteration_count(),
                            # 'ids': train_ids,
                            },
            'val_stats'  : {'stats': val_stats,
                            'niters': val_loader.niters, 'nruns': val_loader.nruns,
                            'totaliters': val_loader.iteration_count(),
                            # 'ids': val_ids,
                            },
            'train_iter' : num_train_iter,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, is_best, savedir=args.save_dir, filename='checkpoint.pth.tar') #_{}.pth.tar'.format(epoch+1))
        print('\n')

    # Delete train and val data loaders
    del train_loader, val_loader

    # Load best model for testing (not latest one)
    print("=> loading best model from '{}'".format(args.save_dir + "/model_best.pth.tar"))
    checkpoint = torch.load(args.save_dir + "/model_best.pth.tar")
    num_train_iter = checkpoint['train_iter']
    try:
        model.load_state_dict(checkpoint['state_dict'])  # BWDs compatibility (TODO: remove)
    except:
        model.load_state_dict(checkpoint['model_state_dict'])
    print("=> loaded best checkpoint (epoch {}, train iter {})"
          .format(checkpoint['epoch'], num_train_iter))
    best_epoch = checkpoint['best_epoch'] if hasattr(checkpoint, 'best_epoch') else 0
    print('==== Best validation loss: {} was from epoch: {} ===='.format(checkpoint['best_loss'],
                                                                         best_epoch))

    # Do final testing (if not asked to evaluate)
    # (don't create the data loader unless needed, creates 4 extra threads)
    print('==== Evaluating trained network on test data ====')
    args.imgdisp_freq = 10 * args.disp_freq # Tensorboard log frequency for the image data
    sampler = torch.utils.data.dataloader.SequentialSampler(test_dataset)  # Run sequentially along the test dataset
    test_loader = DataEnumerator(util.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, sampler=sampler, pin_memory=args.use_pin_memory,
                                    collate_fn=test_dataset.collate_batch))
    test_stats = iterate(test_loader, model, tblogger, len(test_loader),
                         mode='test', epoch=args.epochs)
    print('==== Best validation loss: {} was from epoch: {} ===='.format(best_val_loss,
                                                                         best_epoch))

    # Save final test error
    save_checkpoint({
        'args': args,
        'test_stats': {'stats': test_stats,
                       'niters': test_loader.niters, 'nruns': test_loader.nruns,
                       'totaliters': test_loader.iteration_count(),
                    #    'ids': test_stats.data_ids,
                       },
    }, False, savedir=args.save_dir, filename='test_stats.pth.tar')

    # Close log file
    logfile.close()

################# HELPER FUNCTIONS

### Main iterate function (train/test/val)
def iterate(data_loader, model, tblogger, num_iters,
            mode='test', optimizer=None, epoch=0):
    # Get global stuff?
    global num_train_iter

    # Setup avg time & stats:
    data_time, fwd_time, bwd_time, viz_time  = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Save all stats into a namespace
    stats = argparse.Namespace()
    stats.loss, stats.flowloss                  = AverageMeter(), AverageMeter()
    stats.flowerr_sum, stats.flowerr_avg        = AverageMeter(), AverageMeter()
    stats.motionerr_sum, stats.motionerr_avg    = AverageMeter(), AverageMeter()
    stats.stillerr_sum, stats.stillerr_avg      = AverageMeter(), AverageMeter()
    stats.data_ids = []
    if mode == 'test':
        # Save the flow errors and poses if in "testing" mode
        stats.motion_err, stats.motion_npt, stats.still_err, stats.still_npt = [], [], [], []
        # stats.poses, stats.predmasks, stats.masks = [], [], []
        # stats.gtflows, stats.predflows = [], []
        # stats.pts = []

    # Switch model modes
    train = (mode == 'train')
    if train:
        assert (optimizer is not None), "Please pass in an optimizer if we are iterating in training mode"
        model.train()
    else:
        assert (mode == 'test' or mode == 'val'), "Mode can be train/test/val. Input: {}"+mode
        model.eval()

    # Run an epoch
    print('========== Mode: {}, Starting epoch: {}, Num iters: {} =========='.format(
        mode, epoch, num_iters))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pt_wt = args.pt_wt * args.loss_scale
    itr_cnts = 0
    for i in xrange(num_iters):
        # ============ Load data ============#
        # Start timer
        start = time.time()

        # Get a sample
        sample = iter(data_loader).next()
        itr_cnts += 1
        # stats.data_ids.append(sample['id'].clone())

        # Get inputs and targets (as variables)
        # Currently batchsize is the outer dimension
        pts       = util.req_grad(sample['init_state'].to(device), train)  # Need gradients
        ctrls     = util.req_grad(sample['action'].to(device), train)  # Need gradients
        fwdflows  = util.req_grad(sample['scene_flow'].to(device), False)  # No gradients
        # fwdvis    = util.req_grad(sample['fwdvisibilities'].float().to(device), False)

        # Get jt angles
        # jtangles = util.req_grad(sample['actctrlconfigs'].to(device), train)  # [:, :, args.ctrlids_in_state].type(deftype), requires_grad=train)

        # Measure data loading time
        data_time.update(time.time() - start)

        # ============ FWD pass + Compute loss ============#
        # Start timer
        start = time.time()

        ### Run a FWD pass through the network
        # Make flow prediction
        deltaposes, masks = [], []
        if args.use_se3_nets:
            flows, [deltapose, mask] = model([pts, None, ctrls],
                                             train_iter=num_train_iter)
            deltaposes.append(deltapose)
            masks.append(mask)
        else:
            flows = model([pts[:, 0], jtangles[:, 0], ctrls[:, 0]])
        nextpts = pts[:, :3, ...] + flows  # Add flow to get next prediction

        # Append to list of predictions
        predflows = [flows,]
        predpts = [nextpts,]

        # Get loss function inputs and targets
        inputs = flows  # Predicted flow for that step (note that gradients only go to the mask & deltas)
        targets = fwdflows[:, :3, ...]

        ### 3D loss
        # If motion-normalized loss, pass in GT flows
        if args.motion_norm_loss:
            motion = targets  # Use either delta-flows or full-flows
            loss = pt_wt * ctrlnets.MotionNormalizedLoss3D(inputs, targets, motion=motion,
                                                           loss_type=args.loss_type, wts=torch.Tensor([1.]).to(device))
        else:
            loss = pt_wt * ctrlnets.Loss3D(inputs, targets, loss_type=args.loss_type, wts=1)
        flowloss = torch.Tensor([loss.item()])

        # Update stats
        stats.flowloss.update(flowloss)
        stats.loss.update(loss.item())

        # Measure FWD time
        fwd_time.update(time.time() - start)

        # ============ Gradient backpass + Optimizer step ============#
        # Compute gradient and do optimizer update step (if in training mode)
        if (train):
            # Start timer
            start = time.time()

            # Backward pass & optimize
            optimizer.zero_grad() # Zero gradients
            loss.backward()       # Compute gradients - BWD pass
            optimizer.step()      # Run update step

            # Increment number of training iterations by 1
            num_train_iter += 1

            # Measure BWD time
            bwd_time.update(time.time() - start)

        # ============ Visualization ============#
        # Make sure to not add to the computation graph (will memory leak otherwise)!
        with torch.no_grad():

            # Start timer
            start = time.time()

            # Compute flow predictions and errors
            # NOTE: I'm using CUDA here to speed up computation by ~4x
            predflows_t = torch.cat([(x - pts[:,:3,...]).unsqueeze(1) for x in predpts], 1)
            flows_t = fwdflows
            if args.use_only_da_for_flows:
                # If using only DA then pts that are not visible will not have GT flows, so we shouldn't take them into
                # account when computing the flow errors
                flowerr_sum, flowerr_avg, \
                    motionerr_sum, motionerr_avg,\
                    stillerr_sum, stillerr_avg,\
                    motion_err, motion_npt,\
                    still_err, still_npt         = helpers.compute_masked_flow_errors(predflows_t, flows_t) # Zero out flows for non-visible points
            else:
                flowerr_sum, flowerr_avg, \
                    motionerr_sum, motionerr_avg, \
                    stillerr_sum, stillerr_avg, \
                    motion_err, motion_npt, \
                    still_err, still_npt         = helpers.compute_masked_flow_errors(predflows_t, flows_t)

            # Update stats
            stats.flowerr_sum.update(flowerr_sum); stats.flowerr_avg.update(flowerr_avg)
            stats.motionerr_sum.update(motionerr_sum); stats.motionerr_avg.update(motionerr_avg)
            stats.stillerr_sum.update(stillerr_sum); stats.stillerr_avg.update(stillerr_avg)
            if mode == 'test':
                stats.motion_err.append(motion_err); stats.motion_npt.append(motion_npt)
                stats.still_err.append(still_err); stats.still_npt.append(still_npt)
                # if args.use_se3_nets:
                #     stats.predmasks.append(torch.cat([x.cpu().float().unsqueeze(1) for x in masks]))
                #     stats.masks.append(sample['masks'][:, 0])
                # stats.predflows.append(predflows_t.cpu())
                # stats.gtflows.append(flows_t.cpu())
                # stats.pts.append(sample['points'][:,0])

            # Display/Print frequency
            bsz = pts.size(0)
            if i % args.disp_freq == 0:
                ### Print statistics
                print_stats(mode, epoch=epoch, curr=i+1, total=num_iters,
                            samplecurr=i+1, sampletotal=len(data_loader),
                            stats=stats, bsz=bsz)

                ### Print time taken
                print('\tTime => Data: {data.val:.3f} ({data.avg:.3f}), '
                            'Fwd: {fwd.val:.3f} ({fwd.avg:.3f}), '
                            'Bwd: {bwd.val:.3f} ({bwd.avg:.3f}), '
                            'Viz: {viz.val:.3f} ({viz.avg:.3f})'.format(
                        data=data_time, fwd=fwd_time, bwd=bwd_time, viz=viz_time))

                ### TensorBoard logging
                # (1) Log the scalar values
                iterct = itr_cnts # Get total number of iterations so far
                info = {
                    mode+'-loss': loss.item(),
                    mode+'-flowloss': flowloss.sum(),
                    mode+'-flowerrsum': flowerr_sum.sum()/bsz,
                    mode+'-flowerravg': flowerr_avg.sum()/bsz,
                    mode+'-motionerrsum': motionerr_sum.sum()/bsz,
                    mode+'-motionerravg': motionerr_avg.sum()/bsz,
                    mode+'-stillerrsum': stillerr_sum.sum() / bsz,
                    mode+'-stillerravg': stillerr_avg.sum() / bsz,
                }
                for tag, value in info.items():
                    tblogger.scalar_summary(tag, value, iterct)

                # (2) Log images & print pretdicted SE3s
                # TODO: Numpy or matplotlib
                if i % args.imgdisp_freq == 0 and False:

                    ## Log the images (at a lower rate for now)
                    id = random.randint(0, sample['init_state'].size(0)-1)

                    # Concat the flows, depths and masks into one tensor
                    # flowdisp  = torchvision.utils.make_grid(torch.cat([flows_t.narrow(0,id,1),
                    #                                                    predflows_t.narrow(0,id,1)], 0).cpu().view(-1, 3, args.img_ht, args.img_wd),
                    #                                         nrow=args.seq_len, normalize=True, range=(-0.01, 0.01))
                    # depthdisp = torchvision.utils.make_grid(sample['init_state'][id].narrow(1,2,1), normalize=True, range=(0.0,3.0))

                    # Show as an image summary
                    info = {mode + '-depths': util.to_np(depthdisp.unsqueeze(0)),
                            mode + '-flows': util.to_np(flowdisp.unsqueeze(0)),
                            }
                    if args.use_se3_nets:
                        maskdisp = torchvision.utils.make_grid(
                            torch.cat([masks[0].narrow(0, id, 1)], 0).cpu().view(-1, 1, args.img_ht, args.img_wd),
                            nrow=args.num_se3, normalize=True, range=(0, 1))
                        info[mode+ '-masks'] = util.to_np(maskdisp.narrow(0,0,1))
                    for tag, images in info.items():
                        tblogger.image_summary(tag, images, iterct)

            # Measure viz time
            viz_time.update(time.time() - start)

    ### Print stats at the end
    print('========== Mode: {}, Epoch: {}, Final results =========='.format(mode, epoch))
    print_stats(mode, epoch=epoch, curr=num_iters, total=num_iters,
                samplecurr=data_loader.niters+1, sampletotal=len(data_loader),
                stats=stats)
    print('========================================================')

    # Return the loss & flow loss
    return stats

### Print statistics
def print_stats(mode, epoch, curr, total, samplecurr, sampletotal,
                stats, bsz=None):
    # Print loss
    bsz = args.batch_size if bsz is None else bsz
    print('Mode: {}, Epoch: [{}/{}], Iter: [{}/{}], Sample: [{}/{}], Batch size: {}, '
          'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
        mode, epoch, args.epochs, curr, total, samplecurr,
        sampletotal, bsz, loss=stats.loss))

    # Print flow loss per timestep
    for k in xrange(args.seq_len):
        print('\tStep: {}, Loss: {:.3f} ({:.3f}), '
              'Flow => Sum: {:.3f} ({:.3f}), Avg: {:.3f} ({:.3f}), '
              'Motion/Still => Sum: {:.3f}/{:.3f}, Avg: {:.3f}/{:.3f}'
            .format(
            1 + k * args.step_len,
            stats.flowloss.val[k], stats.flowloss.avg[k],
            stats.flowerr_sum.val[k] / bsz, stats.flowerr_sum.avg[k] / bsz,
            stats.flowerr_avg.val[k] / bsz, stats.flowerr_avg.avg[k] / bsz,
            stats.motionerr_sum.avg[k] / bsz, stats.stillerr_sum.avg[k] / bsz,
            stats.motionerr_avg.avg[k] / bsz, stats.stillerr_avg.avg[k] / bsz,
        ))

### Save checkpoint
def save_checkpoint(state, is_best, savedir='.', filename='checkpoint.pth.tar'):
    savefile = savedir + '/' + filename
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, savedir + '/model_best.pth.tar')

### Adjust learning rate
def adjust_learning_rate(optimizer, epoch, decay_rate=0.1, decay_epochs=10):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (decay_rate ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

################ RUN MAIN
if __name__ == '__main__':
    main()

