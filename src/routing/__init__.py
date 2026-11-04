"""Routing package: pick a language-specific pipeline before fusion."""

from .language_router import LanguageRouter, RouteConfig, RoutedRuleResult

__all__ = ["LanguageRouter", "RouteConfig", "RoutedRuleResult"]
