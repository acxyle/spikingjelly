import argparse
import copy
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
import warnings

import torch
from torch import nn

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

import training_script as legacy
from input_encoder_module import (
    BurstEncoder, IdentityEncoder, LatencyEncoder, PhaseEncoder,
    PoissonEncoder, PopulationEncoder, QuantizationEncoder, RepeatEncoder,
)
from model_wrapper import ModelWrapper, TemporalMeanDecoder, expand_input_channels
from training_script_new import SResNetTrainer, parse_args
from spikingjelly.activation_based import functional, layer, neuron
from tv_ref_classify.utils import ExponentialMovingAverage


torch.set_num_threads(1)


def tiny_backbone():
    model = nn.Sequential(
        layer.Conv2d(3, 4, 1), neuron.LIFNode(), layer.Flatten(), layer.Linear(64, 6),
    )
    functional.set_step_mode(model, "m")
    return model


def wrap(backbone):
    return ModelWrapper(RepeatEncoder(T=2), backbone, TemporalMeanDecoder())


class EncoderTests(unittest.TestCase):
    def test_legacy_equivalence(self):
        pairs = [
            *[(PopulationEncoder, legacy.population_encode_image, {"population_size": p})
              for p in (2, 3, 4, 5, 8)],
            (LatencyEncoder, legacy.latency_encode_image, {}),
            *[(LatencyEncoder, legacy.latency_encode_image, {"mode": mode, "zero_silent": silent})
              for mode in ("linear", "log") for silent in (False, True)],
            (PoissonEncoder, legacy.poisson_encode_image, {}),
            (QuantizationEncoder, legacy.quantization_encode_image, {}),
            (BurstEncoder, legacy.burst_encode_image, {}),
            (BurstEncoder, legacy.burst_encode_image, {"max_spikes": 3, "isi_min": 2}),
            (PhaseEncoder, legacy.phase_encode_image, {}),
            (PhaseEncoder, legacy.phase_encode_image, {"P": 8, "tau": 3.0}),
        ]
        for dtype in (torch.float32, torch.float64):
            pixels = torch.linspace(0, 1, 60, dtype=dtype).reshape(2, 3, 2, 5)
            mean = torch.tensor((.485, .456, .406), dtype=dtype).view(1, 3, 1, 1)
            std = torch.tensor((.229, .224, .225), dtype=dtype).view(1, 3, 1, 1)
            for normalized in (False, True):
                x = (pixels - mean) / std if normalized else pixels
                for cls, function, kwargs in pairs:
                    with self.subTest(encoder=cls.__name__, dtype=dtype, normalized=normalized, kwargs=kwargs):
                        encoder = cls(T=64, **kwargs)
                        torch.manual_seed(100)
                        expected = function(x, T=64, **kwargs)
                        for training in (True, False, True):
                            encoder.train(training)
                            torch.manual_seed(100)
                            actual = encoder(x)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            self.assertEqual(actual.dtype, x.dtype)

    def test_quantization_and_latency_entry_points(self):
        trainer = legacy.SResNetTrainer()
        x = torch.tensor([0., .1, .25, .5, .75, 1.]).reshape(1, 1, 1, -1)
        cases = [("quantization", [], {})]
        cases += [("latency", ["--latency-mode", mode] + ["--latency-zero-silent"] * silent,
                   {"mode": mode, "zero_silent": silent})
                  for mode in ("linear", "log") for silent in (False, True)]
        for name, options, kwargs in cases:
            with self.subTest(encoder=name, options=options):
                argv = ["--encoder", name, "--T", "16"] + options
                args = trainer.get_args_parser().parse_args(argv)
                module_trainer, module_args = parse_args(argv)
                encoder = module_trainer.load_encoder(module_args)
                actual = encoder(x)
                for hook in (trainer.preprocess_train_sample, trainer.preprocess_test_sample):
                    torch.testing.assert_close(actual, hook(args, x), rtol=0, atol=0)
                self.assertTrue(((actual == 0) | (actual == 1)).all())
                counts = actual.sum(0)
                if name == "quantization":
                    torch.testing.assert_close(counts, torch.round(x * 16), rtol=0, atol=0)
                else:
                    expected_counts = torch.ones_like(x)
                    expected_counts[..., 0] = 1 - kwargs["zero_silent"]
                    torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
                    self.assertIn(f"mode={kwargs['mode']}", repr(encoder))
                    self.assertIn(f"zero_silent={kwargs['zero_silent']}", repr(encoder))
        linear = LatencyEncoder(T=16)(x).argmax(0)
        logarithmic = LatencyEncoder(T=16, mode="log")(x).argmax(0)
        self.assertFalse(torch.equal(linear, logarithmic))

    def test_phase_automatic_period_entry_points(self):
        trainer = legacy.SResNetTrainer()
        x = torch.tensor([0., .2, .5, 1.]).reshape(1, 1, 1, -1)
        for T in (32, 64, 128):
            with self.subTest(T=T):
                argv = ["--encoder", "phase", "--T", str(T), "--arch", "sew_resnet18"]
                args = trainer.get_args_parser().parse_args(argv)
                module_trainer, module_args = parse_args(argv)
                encoder = module_trainer.load_encoder(module_args)
                self.assertEqual(encoder.P, T // 4)
                expected = legacy.phase_encode_image(x, T, P=T // 4)
                torch.testing.assert_close(legacy.phase_encode_image(x, T), expected, rtol=0, atol=0)
                torch.testing.assert_close(encoder(x), expected, rtol=0, atol=0)
                torch.testing.assert_close(trainer.encode_sample(args, x), expected, rtol=0, atol=0)
                # Model validation must resolve None before checking the period.
                trainer.load_model(args, num_classes=6)
                self.assertEqual(args.phase_period, T // 4)
                torch.testing.assert_close(trainer.encode_sample(args, x), expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            PhaseEncoder(T=30)
        with self.assertRaises(ValueError):
            legacy.phase_encode_image(x, T=30)
        explicit = PhaseEncoder(T=30, P=5)
        torch.testing.assert_close(explicit(x), legacy.phase_encode_image(x, T=30, P=5), rtol=0, atol=0)

    def test_phase_startup_and_burst_warning(self):
        x = torch.tensor([0., 1.]).view(2, 1, 1, 1)
        encoder = PhaseEncoder()
        result = encoder(x).flatten(1)
        self.assertEqual(result[:, 0].sum(), 0)
        torch.testing.assert_close(result[:, 1], torch.tensor([0.] + [1.] * 15).repeat(4))
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            BurstEncoder(T=8)(x)
        self.assertTrue(any("isi_max >= T" in str(w.message) for w in records))
        with self.assertRaises(ValueError):
            PhaseEncoder(T=4)

    def test_population_state_dtype_and_repr(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            x = torch.linspace(0, 1, 63, dtype=dtype).reshape(1, 3, 3, 7)
            encoder = PopulationEncoder(T=16, population_size=3).to(dtype=dtype)
            expected = legacy.population_encode_image(x, T=16)
            for training in (True, False, True):
                encoder.train(training)
                encoder(torch.ones(2, 3, 1, 1, dtype=dtype))
                actual = encoder(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.shape, (16, 1, 9, 3, 7))
                self.assertTrue(((actual == 0) | (actual == 1)).all())
            self.assertEqual(encoder.state_dict(), {})
            self.assertEqual(list(encoder.parameters()), [])
            self.assertIn("population_size=3", repr(encoder))
            self.assertIn("tau=", repr(encoder))
        self.assertIn("P=16", repr(PhaseEncoder()))
        self.assertIn("isi_max=8", repr(BurstEncoder()))


class CheckpointTests(unittest.TestCase):
    def test_old_weights_optimizer_scheduler_and_ema_resume(self):
        torch.manual_seed(7)
        old = tiny_backbone()
        old_opt = torch.optim.AdamW(old.parameters(), lr=.01)
        old_sched = torch.optim.lr_scheduler.StepLR(old_opt, step_size=1)
        old_ema = ExponentialMovingAverage(old, decay=.9)
        x = torch.rand(2, 3, 4, 4)
        target = torch.tensor([1, 2])

        def step(model, optimizer, wrapped):
            model.train()
            optimizer.zero_grad()
            output = model(x) if wrapped else model(x.unsqueeze(0).repeat(2, 1, 1, 1, 1)).mean(0)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            functional.reset_net(model)
            return output.detach(), loss.detach()

        step(old, old_opt, False)
        old_sched.step()
        old_ema.update_parameters(old)
        checkpoint = copy.deepcopy({
            "model": old.state_dict(), "optimizer": old_opt.state_dict(),
            "lr_scheduler": old_sched.state_dict(), "model_ema": old_ema.state_dict(),
            "args": argparse.Namespace(arch="spiking_resnet18"),
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pth"
            torch.save(checkpoint, path)
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint = torch.load(path, map_location="cpu")

        new = wrap(tiny_backbone())
        new.load_state_dict(checkpoint["model"])
        new_opt = torch.optim.AdamW(new.parameters(), lr=.5)
        new_sched = torch.optim.lr_scheduler.StepLR(new_opt, step_size=1)
        new_opt.load_state_dict(checkpoint["optimizer"])
        new_sched.load_state_dict(checkpoint["lr_scheduler"])
        new_ema = ExponentialMovingAverage(new, decay=.9)
        new_ema.load_state_dict(checkpoint["model_ema"])
        a, loss_a = step(old, old_opt, False)
        b, loss_b = step(new, new_opt, True)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
        old_sched.step()
        new_sched.step()
        self.assertEqual(old_sched.state_dict(), new_sched.state_dict())
        for a, b in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        old_ema.update_parameters(old)
        new_ema.update_parameters(new)
        for key, value in old_ema.module.state_dict().items():
            torch.testing.assert_close(value, new_ema.module.backbone.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(old_ema.n_averaged, new_ema.n_averaged)
        self.assertTrue(all(key.startswith("backbone.") for key in new.state_dict()))
        reloaded = wrap(tiny_backbone())
        reloaded.load_state_dict(new.state_dict())
        new.eval()
        reloaded.eval()
        torch.testing.assert_close(new(x), reloaded(x), rtol=0, atol=0)

    def test_population_roundtrip_and_reject_old_channel_shape(self):
        old = tiny_backbone()
        backbone = tiny_backbone()
        expand_input_channels(backbone, 4)
        self.assertEqual(backbone[0].in_channels, 12)
        self.assertEqual(backbone[0].step_mode, "m")
        new = ModelWrapper(PopulationEncoder(T=2, population_size=4), backbone, TemporalMeanDecoder())
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            new.load_state_dict(old.state_dict())
        clone = copy.deepcopy(new)
        clone.load_state_dict(new.state_dict())
        x = torch.rand(2, 3, 4, 4)
        torch.testing.assert_close(new(x), clone(x), rtol=0, atol=0)

    def test_incomplete_checkpoint_still_fails_strictly(self):
        state = tiny_backbone().state_dict()
        del state["0.weight"]
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            wrap(tiny_backbone()).load_state_dict(state)


class IntegrationTests(unittest.TestCase):
    def test_new_path_does_not_import_old_training_files(self):
        code = '''
import sys
sys.modules["training_script"] = None
sys.modules["train_classify"] = None
from training_script_new import SResNetTrainer, parse_args
trainer, args = parse_args(["--arch", "resnet18", "--encoder", "identity"])
assert type(trainer) is SResNetTrainer
model = trainer.load_model(args, 6)
assert type(model.input_encoder).__name__ == "IdentityEncoder"
'''
        subprocess.run([sys.executable, "-c", code], cwd=MODEL_DIR, check=True, capture_output=True, text=True)

    def test_actual_cnn_backbones(self):
        for arch in ("resnet18", "spiking_resnet18", "sew_resnet18", "vgg11", "spiking_vgg5"):
            with self.subTest(arch=arch):
                trainer, args = parse_args(["--arch", arch, "--T", "2"])
                model = trainer.load_model(args, 6).eval()
                x = torch.rand(2, 3, 32, 32)
                self.assertIs(trainer.preprocess_train_sample(args, x), x)
                self.assertIs(trainer.preprocess_test_sample(args, x), x)
                with torch.no_grad():
                    actual = model(x)
                    functional.reset_net(model)
                    encoded = x.unsqueeze(0).repeat(2, 1, 1, 1, 1)
                    if model.sequence_backbone:
                        expected = model.backbone(encoded).mean(0)
                    else:
                        expected = functional.seq_to_ann_forward(encoded, model.backbone).mean(0)
                self.assertEqual(actual.shape, (2, 6))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                functional.reset_net(model)

    def test_repeat_resnet_backward_and_identity_ann(self):
        trainer, args = parse_args(["--arch", "sew_resnet18", "--encoder", "repeat", "--T", "2"])
        model = trainer.load_model(args, 6)
        self.assertEqual(model.backbone.conv1.in_channels, 3)
        result = model(torch.rand(2, 3, 32, 32))
        result.square().mean().backward()
        self.assertIsNotNone(model.backbone.conv1.weight.grad)
        functional.reset_net(model)
        trainer, args = parse_args(["--arch", "resnet18", "--encoder", "identity"])
        model = trainer.load_model(args, 6).eval()
        x = torch.rand(2, 3, 32, 32)
        self.assertIs(model.input_encoder(x), x)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model.backbone(x), rtol=0, atol=0)

    def test_config_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yml"
            path.write_text("encoder: phase\nT: 64\nphase_period: 16\n")
            trainer, args = parse_args(["-c", str(path), "--encoder", "burst"])
            self.assertEqual(args.encoder, "phase")
            self.assertIsInstance(trainer.load_encoder(args), PhaseEncoder)
            path.write_text("encoder: latency\nlatency_mode: log\nlatency_zero_silent: true\n")
            trainer, args = parse_args(["-c", str(path), "--latency-mode", "linear"])
            encoder = trainer.load_encoder(args)
            self.assertEqual(encoder.mode, "log")
            self.assertTrue(encoder.zero_silent)
            old_parser = argparse.ArgumentParser(add_help=False)
            old_parser.add_argument('-c', '--config')
            old_args = ["training_script.py", "-c", str(path), "--latency-mode", "linear"]
            from unittest.mock import patch
            with patch.object(sys, "argv", old_args):
                args = legacy._parse_args(old_parser, legacy.SResNetTrainer().get_args_parser())
            x = torch.tensor([0., .2, 1.]).reshape(1, 1, 1, -1)
            torch.testing.assert_close(legacy.SResNetTrainer().encode_sample(args, x), encoder(x), rtol=0, atol=0)
        trainer, args = parse_args([])
        self.assertEqual(args.arch, "spiking_resnet18")
        self.assertIsInstance(trainer.load_model(args, 6), ModelWrapper)
        trainer, args = parse_args(["--arch", "spikformer_b_16"])
        with self.assertRaisesRegex(ValueError, "CNN"):
            trainer.load_model(args, 6)
        args.encoder = "population"
        self.assertIsInstance(trainer.load_encoder(args), PopulationEncoder)

    def test_population_resnet_backward_and_legacy_load(self):
        for size in (3, 4):
            trainer, args = parse_args([
                "--arch", "sew_resnet18", "--encoder", "population",
                "--T", "4", "--population-size", str(size),
            ])
            model = trainer.load_model(args, 6)
            self.assertEqual(model.backbone.conv1.in_channels, 3 * size)
            x = torch.rand(2, 3, 32, 32)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            before = model.backbone.conv1.weight.detach().clone()
            model(x).square().mean().backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.backbone.conv1.weight))
            functional.reset_net(model)
            clone = trainer.load_model(args, 6)
            clone.load_state_dict(model.backbone.state_dict())
            model.eval()
            clone.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)
            functional.reset_net(model)
            functional.reset_net(clone)

    def test_real_training_loop_resume_and_validation(self):
        class Dataset(torch.utils.data.Dataset):
            classes = list(range(6))

            def __init__(self):
                self.images = torch.rand(4, 3, 4, 4)

            def __len__(self):
                return 4

            def __getitem__(self, index):
                return self.images[index], index

        class SyntheticData:
            def load_data(self, args):
                dataset = Dataset()
                return dataset, dataset, torch.utils.data.SequentialSampler(dataset), torch.utils.data.SequentialSampler(dataset)

        class OldTrainer(SyntheticData, legacy.SResNetTrainer):
            def load_model(self, args, num_classes):
                return tiny_backbone()

        class NewTrainer(SyntheticData, SResNetTrainer):
            def load_model(self, args, num_classes):
                return wrap(tiny_backbone())

        with tempfile.TemporaryDirectory() as directory:
            _, base_args = parse_args([
                "--arch", "spiking_resnet18", "--T", "2", "--device", "cpu",
                "--disable-amp", "--epochs", "1", "--batch-size", "2", "--workers", "0",
                "--model-ema", "--model-ema-steps", "1", "--lr-warmup-epochs", "0",
            ])
            old_args = copy.deepcopy(base_args)
            old_args.output_dir = str(Path(directory) / "old")
            OldTrainer().main(old_args)
            old_path = next(Path(old_args.output_dir).rglob("checkpoint_latest.pth"))
            new_args = copy.deepcopy(base_args)
            new_args.output_dir = str(Path(directory) / "new")
            new_args.resume = str(old_path)
            new_args.epochs = 2
            NewTrainer().main(new_args)
            new_path = next(Path(new_args.output_dir).rglob("checkpoint_latest.pth"))
            saved = torch.load(new_path, weights_only=False)
            self.assertEqual(saved["epoch"], 1)
            self.assertTrue(all(key.startswith("backbone.") for key in saved["model"]))
            for path in (old_path, new_path):
                args = copy.deepcopy(base_args)
                args.output_dir = str(Path(directory) / ("validation_" + path.parent.parent.parent.name))
                args.resume = str(path)
                args.test_only = True
                NewTrainer().main(args)


if __name__ == "__main__":
    unittest.main()
