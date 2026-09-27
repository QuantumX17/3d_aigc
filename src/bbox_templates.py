# -*- coding: utf-8 -*-
from __future__ import annotations

TEMPLATES = {
    "circular_low_relief": {
        "keywords": ["圆形底座", "圆盘", "浮雕", "桌面浮雕", "低矮圆台", "环绕排列", "浅浮雕"],
        "ratio": (1.00, 1.00, 0.35),
        "default_size_mm": (80.0, 80.0, 28.0),
        "category_hint": {"decorative_object"},
    },
    "upright_plaque": {
        "keywords": ["纪念牌", "竖向浮雕", "名牌底座", "牌雕塑", "铭牌"],
        "ratio": (0.70, 0.38, 1.00),
        "default_size_mm": (60.0, 32.0, 85.0),
        "category_hint": {"decorative_object"},
    },
    "figurine_with_base": {
        "keywords": ["人物", "雕像", "小像", "摆件", "动物", "人物雕塑", "底座"],
        "ratio": (0.55, 0.45, 1.00),
        "default_size_mm": (55.0, 45.0, 100.0),
        "category_hint": {"decorative_object"},
    },
    "vase_like": {
        "keywords": ["花瓶", "瓶", "罐", "容器"],
        "ratio": (0.50, 0.50, 1.00),
        "default_size_mm": (50.0, 50.0, 100.0),
        "category_hint": {"decorative_object", "functional_object"},
    },
    "plate_or_tray": {
        "keywords": ["盘", "托盘", "碟", "浅盘"],
        "ratio": (1.00, 1.00, 0.18),
        "default_size_mm": (100.0, 100.0, 18.0),
        "category_hint": {"functional_object", "decorative_object"},
    },
    "trophy_like": {
        "keywords": ["奖杯", "奖座", "纪念奖座"],
        "ratio": (0.60, 0.50, 1.00),
        "default_size_mm": (60.0, 50.0, 100.0),
        "category_hint": {"decorative_object"},
    },
    "totem_like": {
        "keywords": ["图腾", "柱状", "塔状", "竖直", "细长", "柱体"],
        "ratio": (0.40, 0.40, 1.00),
        "default_size_mm": (40.0, 40.0, 100.0),
        "category_hint": {"decorative_object", "constrained"},
    },
    "wide_functional_object": {
        "keywords": ["支架", "收纳", "托架", "holder", "stand"],
        "ratio": (1.00, 0.70, 0.55),
        "default_size_mm": (100.0, 70.0, 55.0),
        "category_hint": {"functional_object"},
    },
    "generic_decorative": {
        "keywords": [],
        "ratio": (0.75, 0.65, 1.00),
        "default_size_mm": (75.0, 65.0, 100.0),
        "category_hint": {"decorative_object"},
    },
    "generic_functional": {
        "keywords": [],
        "ratio": (1.00, 0.75, 0.65),
        "default_size_mm": (100.0, 75.0, 65.0),
        "category_hint": {"functional_object"},
    },
    "generic_constrained": {
        "keywords": [],
        "ratio": (0.80, 0.70, 1.00),
        "default_size_mm": (80.0, 70.0, 100.0),
        "category_hint": {"constrained"},
    },
}

CATEGORY_DEFAULT_TEMPLATE = {
    "decorative_object": "generic_decorative",
    "functional_object": "generic_functional",
    "constrained": "generic_constrained",
}

FDM_RULES = {
    "stable_base_min_width_ratio_to_height": 0.35,
    "stable_base_min_depth_ratio_to_height": 0.25,
    "min_width_mm": 20.0,
    "min_depth_mm": 12.0,
    "min_height_mm": 8.0,
}
