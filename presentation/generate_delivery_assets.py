#!/usr/bin/env python3
"""Generate the preliminary presentation PPTX and speaker notes DOCX.

This script avoids third-party Python packages so it can run in the current
environment. It builds a minimal PPTX package on top of the Office theme
shipped with Microsoft PowerPoint, and it generates the speaker notes DOCX by
converting HTML with macOS textutil.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape


ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_DIR = ROOT / "presentation"
PPTX_OUT = PRESENTATION_DIR / "ST545_Preliminary_Presentation_Apr1_2026.pptx"
DOCX_OUT = PRESENTATION_DIR / "ST545_Preliminary_Speaker_Notes_Apr1_2026.docx"
THEME_PATH = Path(
    "/Applications/Microsoft PowerPoint.app/Contents/Resources/Office Themes/Office Theme.thmx"
)

SLIDE_W = 12192000
SLIDE_H = 6858000
EMU_PER_INCH = 914400

NAVY = "0E2841"
TEAL = "156082"
ORANGE = "E97132"
GREEN = "196B24"
LIGHT_GRAY = "F3F5F7"
MID_GRAY = "6B7280"
DARK = "1F2937"
WHITE = "FFFFFF"
BORDER = "D0D7DE"


def emu(inches: float) -> int:
    return int(round(inches * EMU_PER_INCH))


def pt(size: int) -> int:
    return size * 100


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")
    return struct.unpack(">II", header[16:24])


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        data = f.read()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"{path} is not a JPEG file")
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            _, height, width = struct.unpack(">BHH", data[i + 4 : i + 9])
            return width, height
        block_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        i += 2 + block_len
    raise ValueError(f"Could not read JPEG dimensions for {path}")


def image_dimensions(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return png_dimensions(path)
    if suffix in {".jpg", ".jpeg"}:
        return jpeg_dimensions(path)
    raise ValueError(f"Unsupported image type: {path}")


def fit_image(path: Path, box_w: int, box_h: int) -> tuple[int, int]:
    img_w, img_h = image_dimensions(path)
    scale = min(box_w / img_w, box_h / img_h)
    return int(img_w * scale), int(img_h * scale)


class SlideComposer:
    def __init__(self) -> None:
        self._shape_id = 2
        self._rel_id = 2
        self.shapes: list[str] = []
        self.relationships: list[tuple[str, str, str]] = [
            (
                "rId1",
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout",
                "../slideLayouts/slideLayout7.xml",
            )
        ]

    def next_shape_id(self) -> int:
        value = self._shape_id
        self._shape_id += 1
        return value

    def next_rel_id(self) -> str:
        value = f"rId{self._rel_id}"
        self._rel_id += 1
        return value

    def add_relationship(self, rel_type: str, target: str) -> None:
        self.relationships.append((self.next_rel_id(), rel_type, target))

    def add_rect(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        fill: str,
        line: str | None = None,
        radius: str = "rect",
    ) -> None:
        sid = self.next_shape_id()
        line_xml = (
            "<a:ln><a:noFill/></a:ln>"
            if line is None
            else f"<a:ln w=\"12700\"><a:solidFill><a:srgbClr val=\"{line}\"/></a:solidFill></a:ln>"
        )
        self.shapes.append(
            f"""
            <p:sp>
              <p:nvSpPr>
                <p:cNvPr id="{sid}" name="Shape {sid}" />
                <p:cNvSpPr />
                <p:nvPr />
              </p:nvSpPr>
              <p:spPr>
                <a:xfrm><a:off x="{x}" y="{y}" /><a:ext cx="{cx}" cy="{cy}" /></a:xfrm>
                <a:prstGeom prst="{radius}"><a:avLst /></a:prstGeom>
                <a:solidFill><a:srgbClr val="{fill}" /></a:solidFill>
                {line_xml}
              </p:spPr>
              <p:txBody><a:bodyPr /><a:lstStyle /><a:p /></p:txBody>
            </p:sp>
            """
        )

    def add_textbox(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        paragraphs: list[str],
        font_size: int = 18,
        color: str = DARK,
        font_face: str = "Aptos",
        bold: bool = False,
        fill: str | None = None,
        line: str | None = None,
        align: str = "l",
        inset: int = emu(0.06),
    ) -> None:
        sid = self.next_shape_id()
        fill_xml = (
            "<a:noFill />"
            if fill is None
            else f"<a:solidFill><a:srgbClr val=\"{fill}\" /></a:solidFill>"
        )
        line_xml = (
            "<a:ln><a:noFill/></a:ln>"
            if line is None
            else f"<a:ln w=\"12700\"><a:solidFill><a:srgbClr val=\"{line}\"/></a:solidFill></a:ln>"
        )
        paras_xml = []
        for para in paragraphs:
            text = xml_escape(para)
            paras_xml.append(
                f"""
                <a:p>
                  <a:pPr algn="{align}" />
                  <a:r>
                    <a:rPr lang="en-US" sz="{pt(font_size)}" {'b="1"' if bold else ''}>
                      <a:solidFill><a:srgbClr val="{color}" /></a:solidFill>
                      <a:latin typeface="{font_face}" />
                    </a:rPr>
                    <a:t>{text}</a:t>
                  </a:r>
                  <a:endParaRPr lang="en-US" sz="{pt(font_size)}" />
                </a:p>
                """
            )
        self.shapes.append(
            f"""
            <p:sp>
              <p:nvSpPr>
                <p:cNvPr id="{sid}" name="TextBox {sid}" />
                <p:cNvSpPr txBox="1" />
                <p:nvPr />
              </p:nvSpPr>
              <p:spPr>
                <a:xfrm><a:off x="{x}" y="{y}" /><a:ext cx="{cx}" cy="{cy}" /></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst /></a:prstGeom>
                {fill_xml}
                {line_xml}
              </p:spPr>
              <p:txBody>
                <a:bodyPr wrap="square" anchor="t" lIns="{inset}" tIns="{inset}" rIns="{inset}" bIns="{inset}" />
                <a:lstStyle />
                {''.join(paras_xml)}
              </p:txBody>
            </p:sp>
            """
        )

    def add_picture(
        self,
        image_path: Path,
        x: int,
        y: int,
        box_w: int,
        box_h: int,
        media_name: str,
    ) -> None:
        display_w, display_h = fit_image(image_path, box_w, box_h)
        x_off = x + (box_w - display_w) // 2
        y_off = y + (box_h - display_h) // 2
        sid = self.next_shape_id()
        rid = self.next_rel_id()
        self.relationships.append(
            (
                rid,
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                f"../media/{media_name}",
            )
        )
        self.shapes.append(
            f"""
            <p:pic>
              <p:nvPicPr>
                <p:cNvPr id="{sid}" name="Picture {sid}" />
                <p:cNvPicPr><a:picLocks noChangeAspect="1" /></p:cNvPicPr>
                <p:nvPr />
              </p:nvPicPr>
              <p:blipFill>
                <a:blip r:embed="{rid}" />
                <a:stretch><a:fillRect /></a:stretch>
              </p:blipFill>
              <p:spPr>
                <a:xfrm><a:off x="{x_off}" y="{y_off}" /><a:ext cx="{display_w}" cy="{display_h}" /></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst /></a:prstGeom>
                <a:ln w="12700"><a:solidFill><a:srgbClr val="{BORDER}" /></a:solidFill></a:ln>
              </p:spPr>
            </p:pic>
            """
        )

    def build_xml(self) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name="" />
        <p:cNvGrpSpPr />
        <p:nvPr />
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0" />
          <a:ext cx="0" cy="0" />
          <a:chOff x="0" y="0" />
          <a:chExt cx="0" cy="0" />
        </a:xfrm>
      </p:grpSpPr>
      {''.join(self.shapes)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping /></p:clrMapOvr>
</p:sld>
"""

    def build_rels_xml(self) -> str:
        rels = []
        for rid, rel_type, target in self.relationships:
            rels.append(
                f'<Relationship Id="{rid}" Type="{rel_type}" Target="{target}" />'
            )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rels)
            + "</Relationships>"
        )


SLIDES = [
    {
        "kind": "title",
        "title": "Predictive Modeling of Equity Trends via Data-Driven\nMedia Weighting and Nonlinear Interaction Analysis",
        "subtitle": [
            "Jacky Luo",
            "ST 545 Modern Statistical Learning",
            "Preliminary project presentation | April 1, 2026",
            "Goal: predict next-day stock direction from market data, fundamentals, and financial news",
        ],
    },
    {
        "kind": "bullets",
        "title": "Motivation and Research Questions",
        "bullets": [
            "Learn publisher-specific weights instead of assigning source importance by hand",
            "Compare FinBERT against TF-IDF for financial text signal extraction",
            "Test nonlinear interaction effects between news sentiment and fundamentals",
            "Use SHAP to separate sentiment signal from market and technical signal",
        ],
    },
    {
        "kind": "image_right",
        "title": "Data Sources and Modeling Target",
        "bullets": [
            "Alpaca: daily OHLCV bars and Benzinga news",
            "Finnhub: PE fundamentals and multi-source company news",
            "10 tickers: NVDA, GOOGL, MSFT, AMZN, TSLA, LMT, NEM, AAPL, META, JPM",
            "Current cache: 46,461 articles and 2,390 merged daily rows",
            "Prediction target: day t features -> day t+1 direction",
        ],
        "image": ROOT / "poc/result/step1_2/publisher_distribution.png",
        "caption": "Publisher mix in the current news cache",
    },
    {
        "kind": "pipeline",
        "title": "Current Modeling Pipeline",
        "steps": [
            ("1. Data pipeline", "Alpaca + Finnhub"),
            ("2. Text models", "TF-IDF vs FinBERT"),
            ("3. Media weights", "Lasso / Ridge"),
            ("4. Tuned models", "PCA + XGBoost + SHAP"),
            ("5. Latest hybrid", "DQS gated v13"),
        ],
        "footnote": "Latest v13 feature fusion: market features + weighted sentiment stats + PCA(8) + top 10 keywords.",
    },
    {
        "kind": "image_right",
        "title": "Preliminary Results",
        "bullets": [
            "Step 4 mean FinBERT gain over TF-IDF: +0.0063 AUC",
            "Mean SHAP sentiment attribution: 80.16%",
            "Best saved hybrid results: LMT v12 = 0.6036, GOOGL v13 = 0.5971",
            "Strongest standalone feature block so far: keywords-only mean AUC = 0.5564",
            "Takeaway: news contains signal, but fusion remains uneven across tickers",
        ],
        "image": ROOT / "poc/result/step4/representation_comparison_xgb.png",
        "caption": "Tuned NLP battle across the 10-ticker universe",
    },
    {
        "kind": "image_right",
        "title": "What We Have Learned So Far",
        "bullets": [
            "v13 mean hybrid AUC = 0.5362",
            "Market-only = 0.5431; keywords-only = 0.5564",
            "Hybrid beats every single baseline for only one ticker: GOOGL",
            "LMT ablation: MLP 0.6782 > RF 0.5923 > XGBoost 0.5653",
            "Next shift: modular experts or deeper fusion models",
        ],
        "image": ROOT / "poc/result/step4/shap_summary_LMT.png",
        "caption": "LMT SHAP example from the tuned XGBoost path",
    },
    {
        "kind": "two_col",
        "title": "Expected Deliverables and Next Steps",
        "left_title": "Expected deliverables",
        "left_items": [
            "Cleaned multimodal daily dataset",
            "Reproducible end-to-end Python pipeline",
            "Comparative model evaluation",
            "SHAP and ablation analysis",
            "Final report and final presentation",
        ],
        "right_title": "Next steps",
        "right_items": [
            "Test a modular expert strategy by ticker type",
            "Improve fusion so hybrid beats market-only and keyword-only baselines",
            "Package artifacts into a final submission workflow",
            "Optional lightweight dashboard if time permits",
        ],
        "closing": "The project already shows measurable news signal; the final challenge is making the hybrid model stable and consistently better than simpler baselines.",
    },
]


NOTES = [
    (
        "Slide 1. Title",
        [
            "Hi everyone. This project asks a simple question: can we predict next-day stock direction better if we combine market data with financial news, instead of using either one alone?",
            "A second goal is to learn which publishers matter from the data itself, instead of assigning source weights by hand.",
        ],
    ),
    (
        "Slide 2. Motivation and Research Questions",
        [
            "There are really three questions behind the project. First, can publisher importance be learned directly from data?",
            "Second, does FinBERT actually beat a simpler TF-IDF baseline here? Third, when I combine sentiment with market and fundamental variables in a nonlinear model, do I get useful extra signal, and can SHAP still show where that signal is coming from?",
        ],
    ),
    (
        "Slide 3. Data Sources and Modeling Target",
        [
            "The data pipeline is pretty straightforward. Alpaca gives me daily bars and Benzinga news, and Finnhub adds PE fundamentals plus extra news sources like Yahoo and SeekingAlpha.",
            "Right now the cache has 46,461 articles across 10 U.S. tickers, which becomes 2,390 merged daily observations. The prediction task is: use day t information to predict the direction on day t plus 1.",
        ],
    ),
    (
        "Slide 4. Current Modeling Pipeline",
        [
            "The modeling pipeline grew in stages rather than all at once. I started with TF-IDF and FinBERT, then estimated ticker-specific media weights with regularized models.",
            "After that I reduced the FinBERT embeddings with PCA, tuned XGBoost, and used SHAP for interpretation. The latest version, v13, also adds DQS gating so noisy news days get down-weighted before the final hybrid prediction step.",
        ],
    ),
    (
        "Slide 5. Preliminary Results",
        [
            "So far the results are mixed, but they are still useful. FinBERT is only a little better than TF-IDF on average, so better text representation alone does not solve the problem.",
            "At the same time, SHAP says sentiment-related features explain about 80 percent of model importance on average, so news is clearly contributing signal. The best saved hybrid AUC is around 0.60, and the strongest single feature block right now is the keyword-based one.",
        ],
    ),
    (
        "Slide 6. What We Have Learned So Far",
        [
            "The main thing I have learned is that finding signal is easier than fusing signal. In the current v13 results, the full hybrid model is not yet consistently better than the market-only or keyword-only baselines.",
            "But the LMT ablation is encouraging: an MLP reaches 0.6782 AUC and beats the tree-based models by a clear margin. That suggests the next improvement probably needs a more flexible fusion model, or a more modular expert-style approach.",
        ],
    ),
    (
        "Slide 7. Expected Deliverables and Next Steps",
        [
            "By the end of the project, I want to deliver a cleaned multimodal dataset, a reproducible pipeline, a comparative evaluation, and interpretation through SHAP and ablation.",
            "If time permits, I may also package the results into a lightweight dashboard. The main next step is to make the hybrid model consistently outperform the simpler baselines, not just occasionally.",
        ],
    ),
]


def add_content_shell(comp: SlideComposer, title: str, slide_index: int) -> None:
    comp.add_rect(0, 0, SLIDE_W, emu(0.62), NAVY)
    comp.add_rect(0, emu(0.62), SLIDE_W, emu(0.06), TEAL)
    comp.add_textbox(
        emu(0.55),
        emu(0.12),
        emu(8.8),
        emu(0.36),
        [title],
        font_size=26,
        color=WHITE,
        font_face="Aptos Display",
        bold=True,
    )
    comp.add_textbox(
        emu(11.9),
        emu(7.06),
        emu(1.0),
        emu(0.24),
        [f"{slide_index}/{len(SLIDES)}"],
        font_size=10,
        color=MID_GRAY,
        align="r",
    )


def build_title_slide(slide_index: int) -> SlideComposer:
    comp = SlideComposer()
    comp.add_rect(0, 0, SLIDE_W, SLIDE_H, NAVY)
    comp.add_rect(emu(0.8), emu(1.15), emu(4.2), emu(0.10), ORANGE)
    comp.add_rect(emu(0.8), emu(1.33), emu(2.2), emu(0.10), TEAL)
    comp.add_textbox(
        emu(0.95),
        emu(1.65),
        emu(10.9),
        emu(2.0),
        SLIDES[0]["title"].split("\n"),
        font_size=30,
        color=WHITE,
        font_face="Aptos Display",
        bold=True,
    )
    comp.add_textbox(
        emu(0.98),
        emu(4.35),
        emu(8.9),
        emu(1.6),
        SLIDES[0]["subtitle"],
        font_size=18,
        color="DDE6ED",
        font_face="Aptos",
    )
    comp.add_rect(emu(9.95), emu(5.95), emu(2.55), emu(0.62), TEAL, radius="roundRect")
    comp.add_textbox(
        emu(10.08),
        emu(6.05),
        emu(2.25),
        emu(0.40),
        ["5-minute preliminary update"],
        font_size=14,
        color=WHITE,
        font_face="Aptos",
        bold=True,
        align="ctr",
    )
    comp.add_textbox(
        emu(0.95),
        emu(6.88),
        emu(4.2),
        emu(0.22),
        [f"Slide {slide_index}"],
        font_size=10,
        color="A7B7C7",
    )
    return comp


def build_bullets_slide(slide: dict, slide_index: int) -> SlideComposer:
    comp = SlideComposer()
    add_content_shell(comp, slide["title"], slide_index)
    comp.add_textbox(
        emu(0.82),
        emu(1.08),
        emu(11.4),
        emu(5.6),
        [f"- {item}" for item in slide["bullets"]],
        font_size=20,
        color=DARK,
        font_face="Aptos",
    )
    return comp


def build_image_right_slide(slide: dict, slide_index: int, media_name: str) -> SlideComposer:
    comp = SlideComposer()
    add_content_shell(comp, slide["title"], slide_index)
    comp.add_textbox(
        emu(0.78),
        emu(1.04),
        emu(5.85),
        emu(5.15),
        [f"- {item}" for item in slide["bullets"]],
        font_size=18,
        color=DARK,
        font_face="Aptos",
    )
    comp.add_rect(emu(7.0), emu(1.12), emu(5.55), emu(4.32), LIGHT_GRAY, line=BORDER)
    comp.add_picture(
        slide["image"],
        emu(7.08),
        emu(1.20),
        emu(5.39),
        emu(4.00),
        media_name,
    )
    comp.add_textbox(
        emu(7.05),
        emu(5.53),
        emu(5.45),
        emu(0.36),
        [slide["caption"]],
        font_size=11,
        color=MID_GRAY,
        align="ctr",
    )
    return comp


def build_pipeline_slide(slide: dict, slide_index: int) -> SlideComposer:
    comp = SlideComposer()
    add_content_shell(comp, slide["title"], slide_index)
    colors = [TEAL, ORANGE, GREEN, NAVY, "4EA72E"]
    x_positions = [0.55, 2.95, 5.35, 7.75, 10.15]
    for idx, ((heading, sub), color, x_pos) in enumerate(zip(slide["steps"], colors, x_positions), start=1):
        comp.add_rect(emu(x_pos), emu(2.05), emu(2.05), emu(1.02), color, radius="roundRect")
        comp.add_textbox(
            emu(x_pos + 0.10),
            emu(2.18),
            emu(1.85),
            emu(0.70),
            [heading, sub],
            font_size=15,
            color=WHITE,
            font_face="Aptos",
            bold=True,
            align="ctr",
        )
        if idx < len(slide["steps"]):
            comp.add_textbox(
                emu(x_pos + 2.07),
                emu(2.38),
                emu(0.22),
                emu(0.24),
                [">"],
                font_size=18,
                color=MID_GRAY,
                bold=True,
                align="ctr",
            )
    comp.add_rect(emu(0.88), emu(4.15), emu(11.55), emu(1.32), LIGHT_GRAY, line=BORDER, radius="roundRect")
    comp.add_textbox(
        emu(1.08),
        emu(4.42),
        emu(11.10),
        emu(0.80),
        [
            "The current implementation moved from simple sentiment scores to a gated hybrid pipeline.",
            slide["footnote"],
        ],
        font_size=17,
        color=DARK,
        font_face="Aptos",
        align="ctr",
    )
    return comp


def build_two_col_slide(slide: dict, slide_index: int) -> SlideComposer:
    comp = SlideComposer()
    add_content_shell(comp, slide["title"], slide_index)
    comp.add_rect(emu(0.82), emu(1.08), emu(5.55), emu(4.82), LIGHT_GRAY, line=BORDER, radius="roundRect")
    comp.add_rect(emu(6.95), emu(1.08), emu(5.55), emu(4.82), LIGHT_GRAY, line=BORDER, radius="roundRect")
    comp.add_rect(emu(1.02), emu(1.25), emu(2.35), emu(0.42), TEAL, radius="roundRect")
    comp.add_rect(emu(7.15), emu(1.25), emu(1.85), emu(0.42), ORANGE, radius="roundRect")
    comp.add_textbox(
        emu(1.12),
        emu(1.30),
        emu(2.15),
        emu(0.26),
        [slide["left_title"]],
        font_size=14,
        color=WHITE,
        bold=True,
        align="ctr",
    )
    comp.add_textbox(
        emu(7.25),
        emu(1.30),
        emu(1.65),
        emu(0.26),
        [slide["right_title"]],
        font_size=14,
        color=WHITE,
        bold=True,
        align="ctr",
    )
    comp.add_textbox(
        emu(1.05),
        emu(1.82),
        emu(5.05),
        emu(3.65),
        [f"- {item}" for item in slide["left_items"]],
        font_size=17,
        color=DARK,
        font_face="Aptos",
    )
    comp.add_textbox(
        emu(7.18),
        emu(1.82),
        emu(5.05),
        emu(3.65),
        [f"- {item}" for item in slide["right_items"]],
        font_size=17,
        color=DARK,
        font_face="Aptos",
    )
    comp.add_rect(emu(0.95), emu(6.08), emu(11.8), emu(0.62), TEAL, radius="roundRect")
    comp.add_textbox(
        emu(1.12),
        emu(6.18),
        emu(11.40),
        emu(0.34),
        [slide["closing"]],
        font_size=13,
        color=WHITE,
        bold=True,
        align="ctr",
    )
    return comp


def build_slide(slide_index: int, slide: dict, media_name: str | None = None) -> SlideComposer:
    kind = slide["kind"]
    if kind == "title":
        return build_title_slide(slide_index)
    if kind == "bullets":
        return build_bullets_slide(slide, slide_index)
    if kind == "image_right":
        if media_name is None:
            raise ValueError("image_right slide requires media")
        return build_image_right_slide(slide, slide_index, media_name)
    if kind == "pipeline":
        return build_pipeline_slide(slide, slide_index)
    if kind == "two_col":
        return build_two_col_slide(slide, slide_index)
    raise ValueError(f"Unknown slide kind: {kind}")


def content_types_xml(num_slides: int) -> str:
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/notesMasters/notesMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"/>',
    ]
    for idx in range(1, 12):
        overrides.append(
            f'<Override PartName="/ppt/slideLayouts/slideLayout{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        )
    for idx in range(1, num_slides + 1):
        overrides.append(
            f'<Override PartName="/ppt/slides/slide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
        overrides.append(
            f'<Override PartName="/ppt/notesSlides/notesSlide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Default Extension="jpeg" ContentType="image/jpeg"/>'
        '<Default Extension="jpg" ContentType="image/jpeg"/>'
        + "".join(overrides)
        + "</Types>"
    )


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def core_xml() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>ST545 Preliminary Presentation</dc:title>
  <dc:subject>StockMind preliminary project presentation</dc:subject>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:keywords>ST545, StockMind, presentation</cp:keywords>
  <dc:description>Preliminary project presentation for April 1, 2026.</dc:description>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def app_xml(num_slides: int) -> str:
    titles = "".join(f"<vt:lpstr>Slide {idx}</vt:lpstr>" for idx in range(1, num_slides + 1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Macintosh PowerPoint</Application>
  <PresentationFormat>Custom</PresentationFormat>
  <Slides>{num_slides}</Slides>
  <Notes>{num_slides}</Notes>
  <HiddenSlides>0</HiddenSlides>
  <MMClips>0</MMClips>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Slides</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>{num_slides}</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="{num_slides}" baseType="lpstr">{titles}</vt:vector>
  </TitlesOfParts>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0</AppVersion>
</Properties>
"""


def presentation_xml(num_slides: int) -> str:
    sld_ids = []
    for idx in range(1, num_slides + 1):
        sld_ids.append(f'<p:sldId id="{255 + idx}" r:id="rId{idx + 1}"/>')
    notes_master_rid = f"rId{num_slides + 2}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" autoCompressPictures="0">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f'<p:notesMasterIdLst><p:notesMasterId id="2147483649" r:id="{notes_master_rid}"/></p:notesMasterIdLst>'
        f'<p:sldIdLst>{"".join(sld_ids)}</p:sldIdLst>'
        f'<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        '<p:defaultTextStyle>'
        '<a:defPPr><a:defRPr lang="en-US"/></a:defPPr>'
        '<a:lvl1pPr marL="0" algn="l"><a:defRPr sz="1800"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill></a:defRPr></a:lvl1pPr>'
        '</p:defaultTextStyle>'
        '</p:presentation>'
    )


def presentation_rels_xml(num_slides: int) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    for idx in range(1, num_slides + 1):
        rels.append(
            f'<Relationship Id="rId{idx + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{num_slides + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="notesMasters/notesMaster1.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def build_notes_master_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notesMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
               xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
               xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="">
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name="Notes Master" />
        <p:cNvGrpSpPr />
        <p:nvPr />
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0" />
          <a:ext cx="0" cy="0" />
          <a:chOff x="0" y="0" />
          <a:chExt cx="0" cy="0" />
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink" />
</p:notesMaster>
"""


def build_notes_master_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
        "</Relationships>"
    )


def build_notes_slide_xml(paragraphs: list[str]) -> str:
    para_xml = []
    for para in paragraphs:
        para_xml.append(
            f"""
            <a:p>
              <a:r><a:t>{xml_escape(para)}</a:t></a:r>
            </a:p>
            """
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
         xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
         xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="">
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name="" />
        <p:cNvGrpSpPr />
        <p:nvPr />
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0" />
          <a:ext cx="0" cy="0" />
          <a:chOff x="0" y="0" />
          <a:chExt cx="0" cy="0" />
        </a:xfrm>
      </p:grpSpPr>
      <p:sp>
        <p:nvSpPr>
          <p:cNvPr id="3" name="Speaker Notes" />
          <p:cNvSpPr />
          <p:nvPr />
        </p:nvSpPr>
        <p:spPr />
        <p:txBody>
          <a:bodyPr />
          {''.join(para_xml)}
        </p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping /></p:clrMapOvr>
</p:notes>
"""


def build_notes_slide_rels_xml(slide_index: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/slide{slide_index}.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>'
        "</Relationships>"
    )


def copy_theme_parts(zipf: zipfile.ZipFile) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(["unzip", "-q", str(THEME_PATH), "-d", tmpdir], check=True)
        theme_root = Path(tmpdir) / "theme"
        copy_map = {
            theme_root / "slideLayouts": "ppt/slideLayouts",
            theme_root / "slideMasters": "ppt/slideMasters",
        }
        for src_dir, prefix in copy_map.items():
            for path in src_dir.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(src_dir)
                    zipf.write(path, f"{prefix}/{rel.as_posix()}")
        zipf.write(theme_root / "theme" / "theme1.xml", "ppt/theme/theme1.xml")


def generate_pptx() -> None:
    image_map: dict[Path, str] = {}
    with zipfile.ZipFile(PPTX_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("[Content_Types].xml", content_types_xml(len(SLIDES)))
        zipf.writestr("_rels/.rels", root_rels_xml())
        zipf.writestr("docProps/core.xml", core_xml())
        zipf.writestr("docProps/app.xml", app_xml(len(SLIDES)))
        zipf.writestr("ppt/presentation.xml", presentation_xml(len(SLIDES)))
        zipf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels_xml(len(SLIDES)))
        copy_theme_parts(zipf)
        zipf.writestr("ppt/notesMasters/notesMaster1.xml", build_notes_master_xml())
        zipf.writestr("ppt/notesMasters/_rels/notesMaster1.xml.rels", build_notes_master_rels_xml())

        for idx, slide in enumerate(SLIDES, start=1):
            media_name = None
            image_path = slide.get("image")
            if image_path is not None:
                image_path = Path(image_path)
                if image_path not in image_map:
                    image_map[image_path] = f"image{len(image_map) + 1}{image_path.suffix.lower()}"
                    zipf.write(image_path, f"ppt/media/{image_map[image_path]}")
                media_name = image_map[image_path]

            composer = build_slide(idx, slide, media_name)
            composer.add_relationship(
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide",
                f"../notesSlides/notesSlide{idx}.xml",
            )
            zipf.writestr(f"ppt/slides/slide{idx}.xml", composer.build_xml())
            zipf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", composer.build_rels_xml())
            zipf.writestr(f"ppt/notesSlides/notesSlide{idx}.xml", build_notes_slide_xml(NOTES[idx - 1][1]))
            zipf.writestr(
                f"ppt/notesSlides/_rels/notesSlide{idx}.xml.rels",
                build_notes_slide_rels_xml(idx),
            )


def rtf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def build_notes_rtf() -> str:
    chunks = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0 Aptos;}{\f1 Aptos Display;}}",
        r"\paperw12240\paperh15840\margl720\margr720\margt720\margb720",
        r"\viewkind4\uc1",
        r"\pard\sa220\f1\fs56\b ST545 Preliminary Speaker Notes\b0\par",
        r"\pard\sa140\f0\fs24 Project: StockMind | Presentation date: April 1, 2026 | Total target length: about 5 minutes\par",
        r"\pard\sa180\f0\fs24 Suggested pace: about 35 to 45 seconds per slide.\par",
    ]

    for title, paragraphs in NOTES:
        chunks.append(r"\pard\sa220\f1\fs34\b " + rtf_escape(title) + r"\b0\par")
        chunks.append(r"\pard\sa100\f0\fs22\i Suggested pace: 35 to 45 seconds.\i0\par")
        for para in paragraphs:
            chunks.append(r"\pard\sa180\f0\fs24 " + rtf_escape(para) + r"\par")

    chunks.append(r"\pard\sa220\f1\fs34\b Optional shorter version\b0\par")
    chunks.append(
        r"\pard\sa220\f0\fs24 If time runs short during class, shorten Slide 4 and Slide 6 first, and keep the data and results slides unchanged.\par"
    )
    chunks.append("}")
    return "\n".join(chunks)


def generate_docx() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rtf_path = Path(tmpdir) / "speaker_notes.rtf"
        rtf_path.write_text(build_notes_rtf(), encoding="utf-8")
        subprocess.run(
            [
                "textutil",
                "-convert",
                "docx",
                str(rtf_path),
                "-output",
                str(DOCX_OUT),
            ],
            check=True,
        )


def main() -> None:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    generate_pptx()
    generate_docx()
    print(f"Generated: {PPTX_OUT}")
    print(f"Generated: {DOCX_OUT}")


if __name__ == "__main__":
    main()
