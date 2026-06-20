"""Unit tests for visualization theme module."""

import plotly.graph_objects as go
import plotly.io as pio

from panelcast.visualization.theme import COLORBLIND_COLORS, register_themes


class TestColorblindColors:
    """Tests for the COLORBLIND_COLORS palette."""

    def test_palette_has_seven_colors(self):
        assert len(COLORBLIND_COLORS) == 7

    def test_all_colors_are_hex(self):
        for color in COLORBLIND_COLORS:
            assert color.startswith("#")
            assert len(color) == 7

    def test_all_colors_unique(self):
        assert len(set(COLORBLIND_COLORS)) == 7

    def test_blue_is_first(self):
        assert COLORBLIND_COLORS[0] == "#0072B2"

    def test_orange_is_second(self):
        assert COLORBLIND_COLORS[1] == "#E69F00"

    def test_colors_are_valid_hex(self):
        for color in COLORBLIND_COLORS:
            hex_part = color[1:]
            int(hex_part, 16)  # Should not raise


class TestRegisterThemes:
    """Tests for theme registration."""

    def test_light_theme_registered(self):
        register_themes()
        assert "aoty_light" in pio.templates

    def test_dark_theme_registered(self):
        register_themes()
        assert "aoty_dark" in pio.templates

    def test_default_template_is_light(self):
        register_themes()
        assert pio.templates.default == "aoty_light"

    def test_light_theme_has_white_background(self):
        register_themes()
        template = pio.templates["aoty_light"]
        assert template.layout.paper_bgcolor == "white"
        assert template.layout.plot_bgcolor == "white"

    def test_dark_theme_has_dark_background(self):
        register_themes()
        template = pio.templates["aoty_dark"]
        assert template.layout.paper_bgcolor == "#1E1E1E"
        assert template.layout.plot_bgcolor == "#2D2D2D"

    def test_light_theme_uses_colorblind_palette(self):
        register_themes()
        template = pio.templates["aoty_light"]
        assert template.layout.colorway == tuple(COLORBLIND_COLORS)

    def test_dark_theme_uses_colorblind_palette(self):
        register_themes()
        template = pio.templates["aoty_dark"]
        assert template.layout.colorway == tuple(COLORBLIND_COLORS)

    def test_light_theme_uses_serif_font(self):
        register_themes()
        template = pio.templates["aoty_light"]
        assert template.layout.font.family == "serif"

    def test_dark_theme_uses_serif_font(self):
        register_themes()
        template = pio.templates["aoty_dark"]
        assert template.layout.font.family == "serif"

    def test_themes_have_hover_mode(self):
        register_themes()
        for name in ["aoty_light", "aoty_dark"]:
            template = pio.templates[name]
            assert template.layout.hovermode == "x unified"

    def test_light_theme_has_grid(self):
        register_themes()
        template = pio.templates["aoty_light"]
        assert template.layout.xaxis.showgrid is True
        assert template.layout.yaxis.showgrid is True

    def test_dark_theme_font_color(self):
        register_themes()
        template = pio.templates["aoty_dark"]
        assert template.layout.font.color == "#E0E0E0"

    def test_light_theme_font_color(self):
        register_themes()
        template = pio.templates["aoty_light"]
        assert template.layout.font.color == "#333333"

    def test_register_is_idempotent(self):
        register_themes()
        register_themes()
        assert "aoty_light" in pio.templates
        assert "aoty_dark" in pio.templates
