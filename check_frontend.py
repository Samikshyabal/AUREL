"""Optional browser check: start Streamlit first; requires Playwright and Edge."""

import argparse
from io import BytesIO
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

from inference import load_models, predict_all
from test_inference import sample


def check(url: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    image = sample()
    png = BytesIO()
    image.save(png, format="PNG")
    expected = predict_all(load_models(), image)
    with sync_playwright() as runner:
        expect.set_options(timeout=45000)
        browser = runner.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1536, "height": 1400})
        page.set_default_timeout(30000)
        page.goto(url)

        def chapter(prefix, title):
            page.locator('[data-testid="stSidebar"] label').filter(has_text=prefix).click()
            expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible()

        def capture(name):
            expect(page.locator('[data-testid="stException"]')).to_have_count(0)
            page.screenshot(path=output / f"{name}.png", animations="disabled")

        expect(page.locator(".pipeline")).to_be_visible()
        assert page.locator(".pipeline").evaluate("img => img.complete && img.naturalWidth > 0")
        capture("01-overview")
        chapter("Model Intelligence", "Model Intelligence")
        expect(page.get_by_text("11,173,962", exact=True)).to_be_visible()
        capture("02-intelligence")
        chapter("Live Inference Studio", "Live Inference Studio")
        expect(page.get_by_text("Awaiting an image.", exact=False)).to_be_visible()
        expect(page.locator(".prediction")).to_have_count(0)
        upload = page.locator('input[type="file"]')
        upload.set_input_files({"name": "broken.png", "mimeType": "image/png", "buffer": b"not a PNG"})
        expect(page.get_by_text("Cannot read this upload.", exact=False)).to_be_visible()
        upload.set_input_files({"name": "gradient.png", "mimeType": "image/png", "buffer": png.getvalue()})
        rows = page.locator(".prediction")
        expect(rows).to_have_count(3)
        for row, result in zip(rows.all(), expected.values(), strict=True):
            expect(row).to_contain_text(result.predicted_class)
            expect(row).to_contain_text(f"{result.confidence_percent:.2f}%")
        chart = page.locator(".js-plotly-plot")
        expect(chart).to_be_visible()
        traces = chart.evaluate("el => el.data.map(t => ({x: t.x, y: t.y}))")
        for trace, result in zip(traces, expected.values(), strict=True):
            assert trace["y"] == list(result.probabilities)
            assert all(abs(x - 100*p) < 0.0001 for x, p in zip(trace["x"], result.probabilities.values(), strict=True))
        capture("03-studio")
        page.get_by_text("View class probabilities", exact=True).click()
        expect(page.get_by_text("airplane", exact=True).last).to_be_visible()
        print("Studio:", [(v.predicted_class, round(v.confidence_percent, 2)) for v in expected.values()])
        chapter("Performance Lab", "Performance Lab")
        expect(page.get_by_text("Source: accuracy_test (1).csv", exact=False)).to_be_visible()
        expect(chart).to_have_count(1)
        expect(chart).to_be_visible()
        capture("04-performance")
        for metric in ("Test loss", "Parameters", "Recorded latency", "Best validation accuracy"):
            page.get_by_role("combobox").click()
            page.get_by_role("option", name=metric, exact=True).click()
            expect(page.get_by_role("heading", name=metric, exact=True)).to_be_visible()
        figure_data = page.get_by_text("View figure data", exact=True)
        expect(figure_data).to_have_count(1)
        figure_data.click()
        expect(page.get_by_text("90.28", exact=True).last).to_be_visible()
        chapter("Knowledge Transfer Explorer", "Knowledge Transfer Explorer")
        page.get_by_text("Hard targets", exact=True).click()
        expect(page.get_by_text("The ground-truth class", exact=False)).to_be_visible()
        capture("05-transfer")
        chapter("Research Findings", "Research Findings")
        expect(page.get_by_text("90.30% · loss 0.2917", exact=True)).to_be_visible()
        expect(page.get_by_text("90.38% · loss 0.2920", exact=True)).to_be_visible()
        capture("06-findings")

        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(url)
        expect(mobile.locator(".small-flow")).to_be_visible()
        expect(mobile.locator(".pipeline")).to_be_hidden()
        assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth")
        mobile.screenshot(path=output / "01-overview-mobile.png")
        browser.close()
        print("PASS: six chapters, model outputs, comparison controls, upload states and mobile overview.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--output", type=Path, default=Path("docs") / "screenshots")
    args = parser.parse_args()
    check(args.url, args.output)
