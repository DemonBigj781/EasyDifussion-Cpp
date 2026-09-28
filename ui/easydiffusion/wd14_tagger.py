"""Compatibility exports for the native C++ WD14 tagger integration."""

from ui.plugins.server.wd14_tagger.wd14_tagger import WD14TagRequest, tag_image

__all__ = ["WD14TagRequest", "tag_image"]
