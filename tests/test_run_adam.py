try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    # from mock import patch
    exit()
import pytest
import sys

sys.path.append("YoloV3_PyTorch")
from train import *



class TestRun:

    def get_setup_file(self):
        parser = argparse.ArgumentParser(description="Trains the YOLOv3 model.")
        parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg",
                            help="Path to model definition file (.cfg)")
        parser.add_argument("-d", "--data", type=str, default="data/coco.data",
                            help="Path to data config file (.data)")
        parser.add_argument("--hyp", type=str, default="config/hyp.cfg",
                            help="Path to data config file (.data)")
        parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
        parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
        parser.add_argument("--n_cpu", type=int, default=2, help="Number of cpu threads to use during batch generation")
        parser.add_argument("-pw", "--pretrained_weights", type=str,
                            help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
        parser.add_argument("--evaluation_interval", type=int, default=3,
                            help="Interval of epochs between evaluations on validation set")
        parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
        parser.add_argument("--iou_thres", type=float, default=0.1,
                            help="Evaluation: IOU threshold required to qualify as detected")
        parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
        parser.add_argument("--nms_thres", type=float, default=0.3,
                            help="Evaluation: IOU threshold for non-maximum suppression")
        parser.add_argument("--sync_bn", type=int, default=-1,
                            help="Set use of SyncBatchNorm")
        parser.add_argument("--ema", type=int, default=1,
                            help="Set use of ModelEMA")
        parser.add_argument("--scheduler", type=str, default=None,
                            help="Set type of scheduler")
        parser.add_argument("--optimizer", type=str, default=None,
                            help="Set type of optimizer")
        parser.add_argument("--logdir", type=str, default="logs",
                            help="Directory for training log files (e.g. for TensorBoard)")
        parser.add_argument("--name", type=str, default=None,
                            help="Name for trained model")
        parser.add_argument("--warmup", type=bool, default=True,
                            help="Name for trained model")
        parser.add_argument("--clearml", type=bool, default=False,
                            help="Connect to clearml server")
        parser.add_argument("--test_cycle", type=bool, default=False,
                            help="Define if script should return test feedback")
        parser.add_argument("-g", "--gpu", type=int, default=-1, help="Define which gpu should be used")
        parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
        args = parser.parse_args()
        return args

    def parse_hyp_config(path):
        """Parses the hyperparamaters configuration file"""
        options = dict()
        with open(path, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#') or line.startswith('['):
                continue
            key, value = line.split('=')
            options[key.strip()] = value.strip()
        return options

    def parse_data_config(path):
        """Parses the data configuration file"""
        options = dict()
        options['gpus'] = '0,1,2,3'
        options['num_workers'] = '10'
        with open(path, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            key, value = line.split('=')
            options[key.strip()] = value.strip()
        return options

    def test_run_adam_CosineAnnealingLR(self):
        seed = "122"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_CosineAnnealingLR"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'CosineAnnealingLR'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_ExponentialLR(self):
        seed = "126"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_ExponentialLR"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'ExponentialLR'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_ReduceLROnPlateau(self):
        seed = "128"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_ReduceLROnPlateau"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'ReduceLROnPlateau'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_ConstantLR(self):
        seed = "130"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_ConstantLR"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'ConstantLR'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_CyclicLR(self):
        seed = "132"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_CyclicLR"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'CyclicLR'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_OneCycleLR(self):
        seed = "134"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_OneCycleLR"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'OneCycleLR'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_LambdaLR(self):
        seed = "136"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_LambdaLR"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'LambdaLR'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_MultiplicativeLR(self):
        seed = "138"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_MultiplicativeLR"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'MultiplicativeLR'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_StepLR(self):
        seed = "140"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_StepLR"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'StepLR'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_MultiStepLR(self):
        seed = "142"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_MultiStepLR"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'MultiStepLR'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_LinearLR(self):
        seed = "144"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_LinearLR"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'LinearLR'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_PolynomialLR(self):
        seed = "144"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_PolynomialLR"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'PolynomialLR'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"

    def test_run_adam_CosineAnnealingWarmRestarts(self):
        seed = "146"
        epochs = "25"
        gpu = "0"
        name = "test_run_adam_CosineAnnealingWarmRestarts"
        '''
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax","adam"]
        '''
        optimizer = "adam"
        '''
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR','MultiplicativeLR',
                                  'StepLR','MultiStepLR','LinearLR','PolynomialLR','CosineAnnealingWarmRestarts']
        '''
        scheduler = 'CosineAnnealingWarmRestarts'
        testargs = ["prog", "-m", "tests/configs/test_run_v2.cfg", "-d", "tests/configs/Test.data", "-e", epochs, "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g",gpu,"--seed",seed,"--scheduler",scheduler,'--optimizer',optimizer,"--name",name,"--test_cycle","True"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            hyp_config = parse_hyp_config(setup.hyp)
            data_config = parse_data_config(setup.data)
            assert run(setup, data_config, hyp_config,
                       'test_cuda') == f"Finished training for {epochs} epochs, with {optimizer} optimizer and {scheduler} lr sheduler"