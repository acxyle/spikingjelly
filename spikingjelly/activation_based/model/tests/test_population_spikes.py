import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import training_script as legacy
import input_encoder_fn as encoder_fn
from spikingjelly.activation_based import functional

torch.set_num_threads(1)


class PopulationSpikeTests(unittest.TestCase):
    def test_calibration_and_reset(self):
        for T in (16, 32, 64):
            for dtype in (torch.float32, torch.float64):
                x = torch.tensor([0., .25, .5, .75, 1.], dtype=dtype).view(1, 1, 1, 5)
                y = encoder_fn.population_encode_image(x, T)
                self.assertEqual(y.dtype, dtype)
                self.assertTrue(((y == 0) | (y == 1)).all())
                counts = y.sum(0)[0, :, 0]
                self.assertTrue((counts[range(3), [0, 2, 4]] == T).all())
                self.assertTrue((counts[range(3), [2, 1, 2]] == round(T / math.e)).all())
                self.assertTrue((counts[range(3), [4, 0, 0]] == 0).all())
                self.assertEqual(encoder_fn._population_parameters(T)[0], 2.)
                torch.testing.assert_close(y, encoder_fn.population_encode_image(x, T), rtol=0, atol=0)
            grid = torch.linspace(0, 1, 1001).view(1, 1, 1, -1)
            self.assertTrue((encoder_fn.population_encode_image(grid, T).sum((0, 2)) > 0).all())

    def test_layout_and_normalization(self):
        x = torch.linspace(0, 1, 60).view(2, 3, 2, 5)
        y = encoder_fn.population_encode_image(x, 16)
        self.assertEqual(y.shape, (16, 2, 9, 2, 5))
        for c in range(3):
            torch.testing.assert_close(y[:, :, 3*c:3*c+3], encoder_fn.population_encode_image(x[:, c:c+1], 16))
        mean = torch.tensor((.485, .456, .406)).view(1, 3, 1, 1)
        std = torch.tensor((.229, .224, .225)).view(1, 3, 1, 1)
        torch.testing.assert_close(y, encoder_fn.population_encode_image((x-mean)/std, 16), rtol=0, atol=0)

    def test_other_population_sizes(self):
        for P in (2, 4, 5, 8):
            anchors, distances, probes, coverage = encoder_fn._population_geometry(P)
            for T in (16, 32, 64):
                tau, widths = encoder_fn._population_parameters(T, P)
                self.assertEqual(len(widths), P)
                self.assertEqual(tau, 2.)
                for row, target in zip(probes[:2], (T, round(T / math.e))):
                    x = torch.tensor(row).view(1, 1, 1, P)
                    counts = encoder_fn.population_encode_image(x, T, P).sum(0)[0, :, 0]
                    self.assertTrue((counts.diagonal() == target).all())
                grid = torch.linspace(0, 1, 1001).view(1, 1, 1, -1)
                spikes = encoder_fn.population_encode_image(grid, T, P)
                self.assertEqual(spikes.shape, (T, 1, P, 1, 1001))
                self.assertTrue((spikes.sum((0, 2)) > 0).all())
        for P in (True, 0, -1, 2.5):
            with self.assertRaises(ValueError):
                encoder_fn._population_parameters(16, P)

    def test_encoder_parser_and_hooks(self):
        trainer = legacy.SResNetTrainer()
        x = torch.linspace(0, 1, 60).reshape(2, 3, 2, 5)
        cases = [
            ('repeat', [], lambda x: x.unsqueeze(0).repeat(64, 1, 1, 1, 1)),
            ('poisson', [], lambda x: encoder_fn.poisson_encode_image(x, 64)),
            ('latency', [], lambda x: encoder_fn.latency_encode_image(x, 64)),
            ('phase', ['--phase-period', '8', '--phase-tau', '3'],
             lambda x: encoder_fn.phase_encode_image(x, 64, P=8, tau=3.)),
            ('burst', ['--burst-max-spikes', '3', '--burst-isi-min', '2', '--burst-isi-max', '6'],
             lambda x: encoder_fn.burst_encode_image(x, 64, max_spikes=3, isi_min=2, isi_max=6)),
            ('population', ['--population-size', '4'],
             lambda x: encoder_fn.population_encode_image(x, 64, population_size=4)),
        ]
        for name, options, fn in cases:
            args = trainer.get_args_parser().parse_args(['--encoder', name, '--T', '64'] + options)
            torch.manual_seed(42)
            expected = fn(x)
            for hook in (trainer.preprocess_train_sample, trainer.preprocess_test_sample):
                torch.manual_seed(42)
                torch.testing.assert_close(hook(args, x), expected, rtol=0, atol=0)
        for options in (['--encoder', 'phase', '--T', '4'],
                        ['--encoder', 'burst', '--burst-isi-min', '0'],
                        ['--encoder', 'burst', '--burst-max-spikes', '-1']):
            args = trainer.get_args_parser().parse_args(options)
            with self.assertRaises(ValueError):
                trainer.load_model(args, num_classes=6)

    def test_invalid_parameters(self):
        for T in (0, 1, -1, True, 3.5):
            with self.assertRaises(ValueError):
                encoder_fn._population_parameters(T)
        with self.assertRaises(ValueError):
            encoder_fn.population_encode_image(torch.zeros(1, 1, 1, 1), 16, population_size=1)

    def test_resnet_training_and_validation(self):
        trainer = legacy.SResNetTrainer()
        args = trainer.get_args_parser().parse_args(['--arch', 'sew_resnet18', '--encoder', 'population', '--T', '4', '--population-size', '4'])
        model = trainer.load_model(args, num_classes=6)
        self.assertEqual(model.conv1.in_channels, 12)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        x = torch.rand(2, 3, 32, 32)
        model.train()
        y = trainer.process_model_output(args, model(trainer.preprocess_train_sample(args, x)))
        torch.nn.functional.cross_entropy(y, torch.tensor([0, 1])).backward()
        self.assertIsNotNone(model.conv1.weight.grad)
        optimizer.step()
        functional.reset_net(model)
        model.eval()
        with torch.no_grad():
            y = trainer.process_model_output(args, model(trainer.preprocess_test_sample(args, x)))
        self.assertEqual(y.shape, (2, 6))
        self.assertTrue(torch.isfinite(y).all())
        functional.reset_net(model)


if __name__ == '__main__':
    unittest.main()
