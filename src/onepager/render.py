"""Render the liquid-token one-pager from a context dict.

    from onepager.render import render, save
    html = render(context)          # -> self-contained HTML string
    save(context, "outputs/<slug>-one-pager.html")

The output is a single self-contained file: Montserrat and the RV logo are
embedded as base64, so it renders identically offline and when shared.
"""
import base64
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import contract
from utils.helper_functions import report_errors

_ROOT = Path(__file__).resolve().parents[2]      # repo root
_TEMPLATE_DIR = _ROOT / "templates"
_TEMPLATE_NAME = "liquid-one-pager.html.j2"
_ASSETS = Path(__file__).resolve().parent / "assets"
_FONT_CSS = _ASSETS / "montserrat.css"
_LOGO = _ASSETS / "rv_lockup.png"


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        undefined=StrictUndefined,      # referencing an unset var errors, not blanks
        autoescape=False,               # values are trusted, pre-formatted display strings
    )


@report_errors(reraise=True)
def render(context: dict) -> str:
    """Validate ``context`` and return the rendered self-contained HTML."""
    contract.validate(context)
    ctx = dict(context)
    ctx["verdict_color"] = contract.verdict_color(ctx["verdict"])
    ctx["font_css"] = _FONT_CSS.read_text(encoding="utf-8")
    ctx["logo_src"] = _data_uri(_LOGO, "image/png")
    return _env().get_template(_TEMPLATE_NAME).render(**ctx)


@report_errors(reraise=True)
def save(context: dict, out_path) -> Path:
    """Render and write to ``out_path``; returns the Path written."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(context), encoding="utf-8")
    return out
