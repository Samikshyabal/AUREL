import io
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch

from inference import load_model, load_models, predict, predict_all
from preprocessing import CLASSES, preprocess_image


def sample() -> Image.Image:
    """A deterministic, unlabeled RGB gradient; never downloads a dataset."""
    return Image.frombytes("RGB", (32, 32), bytes(
        v for y in range(32) for x in range(32) for v in (8*x, 8*y, 4*(x+y))
    ))


class PipelineTests(unittest.TestCase):
    def check_model(self, name: str, device: str) -> None:
        model = load_model(name, device)
        self.assertFalse(any(m.training for m in model.modules()))
        self.assertEqual(next(model.parameters()).device.type, device)
        image = sample()
        with torch.inference_mode():
            scores = model(preprocess_image(image).to(device))
        self.assertEqual(tuple(scores.shape), (1, 10))
        expected = scores[0].softmax(0).cpu()
        result = predict(model, image)
        self.assertEqual(tuple(result.probabilities), CLASSES)
        values = list(result.probabilities.values())
        self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in values))
        self.assertAlmostEqual(sum(values), 1, places=5)
        torch.testing.assert_close(torch.tensor(values), expected)
        self.assertEqual(result.predicted_class, CLASSES[int(scores.argmax())])
        self.assertAlmostEqual(result.confidence_percent, 100 * expected.max().item(), places=4)

    def test_teacher_cpu(self):
        self.check_model("teacher", "cpu")

    def test_baseline_cpu(self):
        self.check_model("baseline", "cpu")

    def test_distilled_cpu(self):
        self.check_model("distilled", "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_all_cuda(self):
        for name in ("teacher", "baseline", "distilled"):
            with self.subTest(model=name):
                self.check_model(name, "cuda")

    def test_shared_upload(self):
        image = sample()
        with io.BytesIO() as upload:
            image.save(upload, format="PNG")
            upload.seek(0)
            models = load_models("cpu")
            results = predict_all(models, upload)
        self.assertEqual(set(results), {"teacher", "baseline", "distilled"})
        for name, result in results.items():
            self.assertEqual(result, predict(models[name], image))
            print(f"{name}: {result.predicted_class}, {result.confidence_percent:.2f}%")

    def test_upload_preprocessing(self):
        image = Image.new("L", (53, 19), 128)
        batch = preprocess_image(image)
        self.assertEqual(tuple(batch.shape), (1, 3, 32, 32))
        expected = (128/255 - torch.tensor([0.4914, 0.4822, 0.4465])) / torch.tensor(
            [0.2470, 0.2435, 0.2616])
        torch.testing.assert_close(batch, expected.view(1, 3, 1, 1).expand_as(batch))

    def test_invalid_checkpoints_do_not_fall_back(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "student_kd.pt"):
                load_model("distilled", "cpu", root)
            torch.save({"unexpected_key": torch.tensor(1)}, root / "student_kd.pt")
            with self.assertRaisesRegex(ValueError, "Incompatible distilled"):
                load_model("distilled", "cpu", root)


if __name__ == "__main__":
    unittest.main()
