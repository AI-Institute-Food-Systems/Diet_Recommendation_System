"""
Meal generation and RDI-per-kcal portioning.

Generator.generate(n, meal=..., clusters=..., pair_codes=...) takes its random draws from one
seeded NumPy generator (self.rng):
  1. pick (meal, cluster) pair codes and sample z ~ N(0, I);
  2. decode food-presence probabilities with the trained CVAE decoder (training-time
     calibration, then the decoder's pair_allowed_gate mask; optional temperature / top-p);
  3. select foods: probability >= 0.5 among allowed foods with nutrient data, at least
     min_solid_items_by_meal non-beverage items, at most max_beverages beverages;
  4. portion them with _portion_rdi_per_kcal(): serving-size start, signed-log2 refinement
     toward per-kcal RDI targets, then rescaling to the meal's energy target under per-item,
     food-group, beverage, sodium/sugar/saturated-fat and total-gram caps.

Pair codes: dense indices of (meal index, original cluster id); meal index 0/1/2 =
breakfast/lunch/dinner.

Callers: cli/evaluate.py (`make eval`) generates the evaluated meals with a 25/35/40% energy
split; analyses/_common.py builds Generator(model=None), sets _last_meals and calls
_portion_rdi_per_kcal() to re-portion existing food sets (`make sup-evaluation`).
"""

import json
from pathlib import Path

import numpy as np
from typing import Dict, List, Optional


def _load_cluster_names() -> Dict[str, Dict[int, str]]:
    """{meal type: {cluster id: name}} from config/cluster_names.json ({} if unreadable)."""
    try:
        path = Path(__file__).resolve().parents[2] / "config" / "cluster_names.json"
        names = json.loads(path.read_text())
        return {str(mt).lower(): {int(k): v for k, v in d.items()} for mt, d in names.items()}
    except Exception:
        return {}


class Generator:
    """
    Meal generator for a trained multi-meal CVAE with a gated decoder (see the module docstring).
    With model=None only the portioning methods can be used.
    """

    def __init__(
        self,
        model,
        feature_names,
        pair_table,                 # dict[int -> (meal_idx, orig_cluster_id)]
        meal_type_mapping,          # dict[str -> int]
        *,
        serving_grams_vec=None,     # np.array(D,), per-item starting grams; default: _get_realistic_serving_sizes()
        cluster_item_stats=None,    # not used; accepted because cli/evaluate.py passes it
        rng_seed=7,
        # Nutrient-based portioning parameters
        nutrients_per_gram=None,        # np.ndarray (D, P): per-gram densities
        nutrient_names=None,            # list[str] of length P
        rdi_per_kcal=None,              # dict nutrient -> target per kcal (e.g., g/kcal, mg/kcal)
        rdi_daily=None,                 # daily RDI: dict nutrient -> value, or array aligned with nutrient_names
        nutrient_weights=None,          # optional dict for nutrient importance
        meal_energy_split=None,         # dict meal_idx -> fraction of daily kcal
        energy_name="energy_kcal",
        # Beverage capping parameters
        food_to_group=None,             # dict mapping food codes to group names
        max_beverage_grams=600,         # maximum grams per beverage item (fallback)
        max_beverage_grams_by_meal=None,  # dict meal_idx -> max grams per beverage
        beverage_groups=("beverages", "alcoholic_beverages"),  # groups considered beverages/liquids
        bev_energy_frac_max=None,       # maximum fraction of energy from beverages (e.g., 0.3)
        # Subcategory expansion parameters
        expand_by_subcategory=False,     # whether to allow subcategory alternatives
        # Portion control parameters
        max_solid_grams_per_item=300.0,  # maximum grams per solid food item
        # Solid item selection parameters
        min_solid_items_by_meal=None,    # dict meal_idx -> minimum number of non-beverage items
        prefer_diverse_subcats=True,     # try to diversify subcategories when adding items
        # Food name mapping for debug output
        code2name=None,                  # dict mapping food codes to human-readable names
        # Nutrient portioning requirements
        require_rdi_for_nutrient_portioning=True,  # whether to require RDI data for nutrient portioning
        # Group guardrails parameters
        condiment_groups=None,           # set of groups treated as condiments (frozen in energy scaling)
        group_max_grams=None,            # dict group -> max grams per item
        # Debug control
        debug=False                      # whether to print debug information
    ):
        import numpy as np
        self.model = model
        self.feature_names = list(feature_names)
        self.D = len(self.feature_names)
        self.pair_table = dict(pair_table or {})
        self.meal_type_mapping = dict(meal_type_mapping or {})

        # Store food_to_group early so serving sizes can use it
        self.food_to_group = dict(food_to_group or {})
        self.code2name = dict(code2name or {})
        self.debug = bool(debug)
        
        if serving_grams_vec is not None:
            self.serving_grams_vec = np.asarray(serving_grams_vec)
        else:
            # Use realistic serving sizes instead of 100g for everything
            self.serving_grams_vec = self._get_realistic_serving_sizes()
        self.rng = np.random.default_rng(int(rng_seed))
        self._gate = self._get_pair_allowed_gate_or_none()

        # Nutrient-based portioning attributes
        self.nutr_per_g = nutrients_per_gram
        self.nutrient_names = list(nutrient_names or [])
        self.energy_name = energy_name

        # Validate nutrient matrix shape and alignment
        if self.nutr_per_g is not None:
            # Check shape alignment
            if self.nutr_per_g.shape[0] != self.D:
                raise ValueError(
                    f"nutrients_per_gram rows ({self.nutr_per_g.shape[0]}) "
                    f"must match feature_names length ({self.D}). "
                    f"Did you align rows to match feature_names order?"
                )
            
            # Check energy nutrient presence
            if self.energy_name not in self.nutrient_names:
                raise ValueError(
                    f"'{self.energy_name}' missing in nutrient_names. "
                    f"Available nutrients: {self.nutrient_names[:5]}{'...' if len(self.nutrient_names) > 5 else ''}"
                )
            
            # Sanity check: verify nutrients_per_gram is per-gram, not per-100g
            if len(self.nutrient_names) > 0:
                energy_idx = self.nutrient_names.index(self.energy_name)
                energy_per_g = np.nanmean(self.nutr_per_g[:, energy_idx])
                if energy_per_g > 50:  # If >50 kcal/g, likely per-100g data
                    raise ValueError(
                        f"nutrients_per_gram appears to be per-100g data (energy={energy_per_g:.1f} kcal/g). "
                        f"Expected per-gram data with energy ~1-6 kcal/g."
                    )
        # RDI daily values - only build if nutrient-based portioning is being used
        if self.nutr_per_g is not None and self.nutrient_names:
            self.rdi_daily = self._build_rdi_daily_from_files(rdi_daily, nutrient_names)
        else:
            # No nutrient data: portioning falls back to serving sizes
            self.rdi_daily = {}
        # Share of daily energy per meal index. Callers pass their own split
        # (cli/evaluate.py: 0.25/0.35/0.40); this default is not used by the pipeline.
        self.meal_energy_split = meal_energy_split or {0: 0.15, 1: 0.22, 2: 0.25}

        # Energy target must come from RDI data (only for nutrient-based portioning)
        self.require_rdi_for_nutrient_portioning = require_rdi_for_nutrient_portioning
        if self.require_rdi_for_nutrient_portioning and self.nutr_per_g is not None and "energy_kcal" not in self.rdi_daily:
            raise ValueError(
                "energy_kcal must be present in RDI data for nutrient-based portioning. "
                "Set require_rdi_for_nutrient_portioning=False if you won't use it."
            )
        # Set energy targets (only when we truly have RDI energy)
        if self.nutr_per_g is not None and "energy_kcal" in self.rdi_daily:
            daily_energy = float(self.rdi_daily["energy_kcal"])
            self.meal_energy_targets = {
                mi: daily_energy * frac for mi, frac in self.meal_energy_split.items()
            }
        else:
            self.meal_energy_targets = {}  # no nutrient portioning targets available

        # RDI-per-kcal targets and optional weights
        self.rdi_per_kcal = dict(rdi_per_kcal or {})           # e.g., {"protein_g": 0.025, "fiber_g": 0.014, ...}
        self.nutrient_weights = dict(nutrient_weights or {})   # e.g., {"protein_g": 2.0, "fiber_g": 1.0, "sugar_g": 1.0}
        
        # Beverage capping attributes
        self.max_beverage_grams = float(max_beverage_grams)
        self.max_beverage_grams_by_meal = dict(max_beverage_grams_by_meal or {})
        self.bev_energy_frac_max = float(bev_energy_frac_max) if bev_energy_frac_max is not None else None
        
        # --- Group guardrails (small caps + freeze in energy retargeting) ---
        self.condiment_groups = set(condiment_groups or {"sugars","fats_oils","condiments_sauces"})
        self.group_max_grams = dict({
            "sugars": 12.0,              # ~3 tsp
            "fats_oils": 20.0,           # ~1–1.5 tbsp
            "condiments_sauces": 20.0,   # generous spoonfuls
            # optional soft caps for other groups:
            "snacks_sweets": 60.0,       # a reasonable pastry/bar amount
        }, **(group_max_grams or {}))
        
        # Default per-meal beverage caps (g) when none are given
        if not max_beverage_grams_by_meal:
            self.max_beverage_grams_by_meal = {0: 300.0, 1: 350.0, 2: 350.0}  # breakfast/lunch/dinner caps
        if self.bev_energy_frac_max is None:
            self.bev_energy_frac_max = 0.25   # ≤25% of a meal from beverages
        self._bev_mask = None
        if self.food_to_group:
            g = [self.food_to_group.get(str(c), "") for c in self.feature_names]
            # Standard beverage groups with substring/alias matching
            def is_beverage_group(group_name):
                group_lower = str(group_name).lower()
                for bev_group in beverage_groups:
                    bev_lower = str(bev_group).lower()
                    # Exact match or substring match
                    if group_lower == bev_lower or bev_lower in group_lower or group_lower in bev_lower:
                        return True
                return False
            
            bev_mask = np.array([is_beverage_group(x) for x in g], dtype=bool)
            
            # Add fluid dairy items to beverage mask
            fluid_dairy_codes = self._get_fluid_dairy_codes()
            fluid_dairy_mask = np.array([str(c) in fluid_dairy_codes for c in self.feature_names], dtype=bool)
            
            self._bev_mask = bev_mask | fluid_dairy_mask
        
        # Subcategory expansion attributes
        self.expand_by_subcategory = bool(expand_by_subcategory)
        self._subcat_of = None
        self._subcat_index = {}
        if food_to_group:
            self._subcat_of = [food_to_group.get(str(c), None) for c in self.feature_names]
            from collections import defaultdict
            buckets = defaultdict(list)
            for j, cat in enumerate(self._subcat_of):
                buckets[cat].append(j)
            self._subcat_index = {k: np.array(v, dtype=int) for k, v in buckets.items()}

        # Portion control attributes
        self.max_solid_grams_per_item = float(max_solid_grams_per_item)
        
        # Solid item selection attributes
        self.min_solid_items_by_meal = dict(min_solid_items_by_meal or {0: 2, 1: 3, 2: 3})
        self.prefer_diverse_subcats = bool(prefer_diverse_subcats)

        # Build clean rdi_per_kcal_vec after initialization
        self._rdi_per_kcal_vec = self._build_rdi_per_kcal_vec()
        
        # Archetype names (config/cluster_names.json) for the debug lines of _select_items
        self._cluster_names = _load_cluster_names()
        
        # Check for Vitamin D unit mismatches
        if isinstance(self.rdi_daily, dict) and self.nutrient_names:
            has_mcg_in_rdi = "vitamin_d_mcg" in self.rdi_daily
            has_iu_in_rdi = "vitamin_d_iu" in self.rdi_daily
            has_mcg_in_nutrients = "vitamin_d_mcg" in self.nutrient_names
            has_iu_in_nutrients = "vitamin_d_iu" in self.nutrient_names
            
            if (has_mcg_in_rdi ^ has_mcg_in_nutrients) and (has_iu_in_rdi ^ has_iu_in_nutrients):
                import warnings
                warnings.warn("Vitamin D appears under different units (mcg vs IU) between RDI and foods; applying conversion 1 mcg = 40 IU.")
        
        # Create mappable mask for foods with valid nutrient data
        self._mappable_mask = None
        if self.nutr_per_g is not None and self.energy_name in self.nutrient_names:
            eidx = self.nutrient_names.index(self.energy_name)
            nz = (self.nutr_per_g > 0).any(axis=1)
            self._mappable_mask = (self.nutr_per_g[:, eidx] > 0) & nz

        # 1) Assert perfect metadata alignment at construction
        if self._gate is None:
            # model=None: portion-only use (the sensitivity and ablation analyses re-portion
            # existing meals). Decoding needs a gated decoder; portioning does not.
            if self.model is not None:
                raise RuntimeError("Model decoder has no 'pair_allowed_gate' layer. This generator expects a gated decoder.")
        else:
            K_gate, D_gate = self._gate.shape
            if len(self.feature_names) != D_gate:
                raise ValueError(
                    f"feature_names length {len(self.feature_names)} != gate D {D_gate} "
                    f"(feature order must be the training-time order)."
                )

            # pair_table keys must be a subset of 0..K_gate-1 (keep original training indices)
            bad_keys = [k for k in self.pair_table.keys() if (int(k) < 0 or int(k) >= K_gate)]
            if bad_keys:
                raise ValueError(f"pair_table contains out-of-range keys {bad_keys[:5]} for gate rows {K_gate}")

            # (do NOT assert len(pair_table) == K_gate)

            # optional: verify embedding capacity equals gate rows
            try:
                K_embed = int(self.model.decoder.get_layer("cluster_embedding_dec").weights[0].shape[0])
                if K_embed != K_gate:
                    raise ValueError(f"cluster_embedding rows {K_embed} != gate rows {K_gate}")
            except Exception:
                pass  # some models won't have this named layer; that's OK

        # infer z-dim once
        self.zdim = self._infer_zdim()
        
        # Expose the latent size as model.latent_dim (read by the pair check in cli/evaluate.py)
        if getattr(self.model, "latent_dim", None) is None:
            try:
                setattr(self.model, "latent_dim", int(self.zdim))
            except Exception:
                pass

    # ---------- low-level helpers ----------
    @staticmethod
    def _sigmoid_np(x):
        import numpy as np
        x = np.clip(x, -80, 80).astype(np.float32)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _logit_np(p, eps=1e-6):
        import numpy as np
        p = np.clip(p, eps, 1.0 - eps).astype(np.float32)
        logits = np.log(p) - np.log1p(-p)
        return logits

    def _infer_zdim(self):
        """Infer z dimension from model attributes or config."""
        # Try model attributes first
        for attr in ["latent_dim", "z_dim", "zdim"]:
            v = getattr(self.model, attr, None)
            try:
                if v and int(v) > 0:
                    return int(v)
            except Exception:
                pass
        
        # Try config if present
        raw = getattr(self.model, "raw_config", None)
        if isinstance(raw, dict):
            for k in ["latent_dim", "z_dim", "latent_size"]:
                v = raw.get(k)
                if isinstance(v, (int, float)) and v > 0:
                    return int(v)
        
        # Try to infer from encoder if available
        try:
            if hasattr(self.model, "encoder"):
                # Try to get input shape from encoder
                encoder_input = self.model.encoder.input
                if hasattr(encoder_input, 'shape'):
                    # For presence-only models, input might be (batch, features)
                    if len(encoder_input.shape) == 2:
                        return 64  # reasonable default for presence-only
                    elif len(encoder_input.shape) == 3:
                        # Multi-input case, try to infer from first input
                        first_input = encoder_input[0] if isinstance(encoder_input, (list, tuple)) else encoder_input
                        if hasattr(first_input, 'shape') and len(first_input.shape) >= 2:
                            return 64  # reasonable default
        except Exception:
            pass
        
        return 64  # fallback

    def _build_rdi_per_kcal_vec(self, sugar_pct_limit=0.10, sfa_pct_limit=0.10):
        """
        Build per-kcal targets from daily RDI:
        vec[j] = RDI_daily[j] / RDI_daily['energy_kcal']   (units per kcal)

        Then apply policy defaults for sugars/SFA if missing,
        and finally apply any explicit overrides from self.rdi_per_kcal.
        """
        import numpy as np

        name2idx = {n: i for i, n in enumerate(self.nutrient_names)}
        vec = np.zeros(len(self.nutrient_names), dtype=np.float32)

        # --- 1) Auto-derive from daily RDI when available ---
        daily_kcal = float(self.rdi_daily.get("energy_kcal", 0.0)) if isinstance(self.rdi_daily, dict) else 0.0
        if daily_kcal > 0:
            for n, j in name2idx.items():
                v = float(self.rdi_daily.get(n, 0.0))
                if v > 0:
                    vec[j] = v / daily_kcal  # per-kcal density (units follow RDI units)

        # Energy anchor is always exactly 1 (kcal/kcal)
        if "energy_kcal" in name2idx:
            vec[name2idx["energy_kcal"]] = 1.0

        # --- Vitamin D aliasing (mcg <-> IU) ---
        # 1 µg = 40 IU
        if self.debug and self.rdi_daily:
            if "vitamin_d_mcg" in self.rdi_daily:
                print(f"[DEBUG] RDI file has vitamin_d_mcg: {self.rdi_daily['vitamin_d_mcg']} mcg daily")
            if "vitamin_d_iu" in self.rdi_daily:
                print(f"[DEBUG] RDI file has vitamin_d_iu: {self.rdi_daily['vitamin_d_iu']} IU daily")
        
        # Fix obviously wrong Vitamin D values (common data errors)
        if self.rdi_daily:
            if "vitamin_d_iu" in self.rdi_daily:
                rdi_iu = float(self.rdi_daily["vitamin_d_iu"])
                # If RDI is suspiciously low (like 274 IU instead of 800 IU), correct to standard value
                # Standard daily RDI is 20 mcg = 800 IU
                if rdi_iu < 600:  # Less than 75% of expected 800 IU
                    self.rdi_daily["vitamin_d_iu"] = 800.0  # Set to correct value
                    if self.debug:
                        print(f"[DEBUG] Vitamin D correction: {rdi_iu} IU → 800.0 IU (corrected to standard RDI)")
            elif "vitamin_d_mcg" in self.rdi_daily:
                rdi_mcg = float(self.rdi_daily["vitamin_d_mcg"])
                # If RDI is suspiciously high (like 800 mcg instead of 20 mcg), correct to standard value
                if rdi_mcg > 50:  # Much higher than expected 20 mcg
                    self.rdi_daily["vitamin_d_mcg"] = 20.0  # Set to correct value
                    if self.debug:
                        print(f"[DEBUG] Vitamin D correction: {rdi_mcg} mcg → 20.0 mcg (corrected to standard RDI)")
        
        if daily_kcal > 0:
            # RDI in mcg, matrix in IU
            if ("vitamin_d_mcg" in (self.rdi_daily or {})) and ("vitamin_d_mcg" not in name2idx) and ("vitamin_d_iu" in name2idx):
                rdi_mcg = float(self.rdi_daily["vitamin_d_mcg"])
                rdi_iu = rdi_mcg * 40.0
                vec[name2idx["vitamin_d_iu"]] = rdi_iu / daily_kcal
                if self.debug:
                    print(f"[DEBUG] Vitamin D conversion: {rdi_mcg} mcg → {rdi_iu} IU daily → {rdi_iu/daily_kcal:.1f} IU/kcal")
            # RDI in IU, matrix in mcg
            elif ("vitamin_d_iu" in (self.rdi_daily or {})) and ("vitamin_d_mcg" in name2idx) and ("vitamin_d_iu" not in name2idx):
                rdi_iu = float(self.rdi_daily["vitamin_d_iu"])
                rdi_mcg = rdi_iu / 40.0
                vec[name2idx["vitamin_d_mcg"]] = rdi_mcg / daily_kcal
                if self.debug:
                    print(f"[DEBUG] Vitamin D conversion: {rdi_iu} IU → {rdi_mcg} mcg daily → {rdi_mcg/daily_kcal:.1f} mcg/kcal")
            # Both RDI and matrix in mcg (after canonicalization) - convert to IU for proper scaling
            elif ("vitamin_d_mcg" in (self.rdi_daily or {})) and ("vitamin_d_mcg" in name2idx):
                rdi_mcg = float(self.rdi_daily["vitamin_d_mcg"])
                rdi_iu = rdi_mcg * 40.0
                # Store as IU per kcal for proper scaling
                vec[name2idx["vitamin_d_mcg"]] = rdi_iu / daily_kcal
                if self.debug:
                    print(f"[DEBUG] Vitamin D conversion (both mcg): {rdi_mcg} mcg → {rdi_iu} IU daily → {rdi_iu/daily_kcal:.1f} IU/kcal (stored as mcg/kcal)")
            # Both RDI and matrix in IU - use directly
            elif ("vitamin_d_iu" in (self.rdi_daily or {})) and ("vitamin_d_iu" in name2idx):
                rdi_iu = float(self.rdi_daily["vitamin_d_iu"])
                vec[name2idx["vitamin_d_iu"]] = rdi_iu / daily_kcal
                if self.debug:
                    print(f"[DEBUG] Vitamin D (both IU): {rdi_iu} IU daily → {rdi_iu/daily_kcal:.1f} IU/kcal")

        # --- 2) Policy defaults (used only if file RDI didn't provide them) ---
        # sugars_total_g: 10% of kcal  → g/kcal
        if "sugars_total_g" in name2idx and vec[name2idx["sugars_total_g"]] == 0.0:
            vec[name2idx["sugars_total_g"]] = float(sugar_pct_limit) / 4.0
        # added_sugars_g: 10% of kcal → g/kcal (mirror total sugars policy)
        if "added_sugars_g" in name2idx and vec[name2idx["added_sugars_g"]] == 0.0:
            vec[name2idx["added_sugars_g"]] = float(sugar_pct_limit) / 4.0
        # sfa_g: 10% of kcal → g/kcal
        if "sfa_g" in name2idx and vec[name2idx["sfa_g"]] == 0.0:
            vec[name2idx["sfa_g"]] = float(sfa_pct_limit) / 9.0

        # --- 3) Explicit user overrides (take precedence) ---
        for k, v in (self.rdi_per_kcal or {}).items():
            if k in name2idx:
                vec[name2idx[k]] = float(v)

        return vec

    def _build_rdi_daily_from_files(self, rdi_daily, nutrient_names):
        """
        Daily RDI as {nutrient: value}. A dict is filtered to nutrient_names; an array is matched
        to nutrient_names by position (NaN entries are kept).

        Raises:
            ValueError: If no RDI data is provided
        """
        if rdi_daily is None:
            raise ValueError(
                "RDI daily values must be provided from harmonized nutrition data. "
                "Use load_harmonized_nutrition() to load RDI values from files and pass "
                "its RDI_per_day as the rdi_daily parameter."
            )
        
        # Check if nutrient_names is None or empty
        if nutrient_names is None or len(nutrient_names) == 0:
            raise ValueError(
                "nutrient_names must be provided and cannot be None or empty. "
                "This should be set from harmonized nutrition data."
            )
            
        # If rdi_daily is already a dict, filter to only include nutrients present in nutrient_names
        if isinstance(rdi_daily, dict):
            filtered_rdi = {}
            for nutrient, value in rdi_daily.items():
                if nutrient in nutrient_names:
                    filtered_rdi[nutrient] = float(value)
            return filtered_rdi
            
        # If rdi_daily is an array, convert to dict using nutrient_names
        if rdi_daily is not None and hasattr(rdi_daily, '__len__') and len(rdi_daily) == len(nutrient_names):
            return {nutrient: float(value) for nutrient, value in zip(nutrient_names, rdi_daily)}
            
        # If rdi_daily is an array but different length, only use matching nutrients
        if rdi_daily is not None and hasattr(rdi_daily, '__len__'):
            filtered_rdi = {}
            for i, nutrient in enumerate(nutrient_names):
                if i < len(rdi_daily):
                    filtered_rdi[nutrient] = float(rdi_daily[i])
            return filtered_rdi
            
        raise ValueError(f"Invalid RDI data format: {type(rdi_daily)}. Expected dict or array.")

    def _get_cluster_name(self, meal_idx, cluster_id):
        """Get human-readable cluster name for debug output."""
        meal_names = {0: "breakfast", 1: "lunch", 2: "dinner"}
        meal_name = meal_names.get(meal_idx, "breakfast")
        return self._cluster_names.get(meal_name, {}).get(cluster_id, f"Cluster {cluster_id}")
    
    def _get_food_category(self, food_code):
        """Get food category/subcategory from food code for debug output."""
        if not hasattr(self, '_subcat_of') or self._subcat_of is None:
            return "Unknown"
        
        try:
            # Use the *exact* string form used in feature_names
            code_str = str(food_code)
            if code_str in self.feature_names:
                idx = self.feature_names.index(code_str)
                return self._subcat_of[idx] if idx < len(self._subcat_of) else "Unknown"
        except Exception:
            pass
        return "Unknown"
    
    def _group_of_idx(self, j: int) -> str:
        """Get the food group for a given food index."""
        return str(self.food_to_group.get(str(self.feature_names[j]), "")).lower()
    
    def _apply_group_caps_row(self, g_row: np.ndarray) -> np.ndarray:
        """Apply group-based caps to a row of grams."""
        if not self.food_to_group or not self.group_max_grams:
            return g_row
        g = g_row.copy()
        for j in np.where(g > 0)[0]:
            grp = self._group_of_idx(j)
            cap = self.group_max_grams.get(grp)
            if cap is not None:
                g[j] = min(g[j], float(cap))
        return g
    
    def _is_condiment_idx(self, j: int) -> bool:
        """Check if a food index represents a condiment group."""
        return self._group_of_idx(j) in self.condiment_groups
    
    def _get_fluid_dairy_codes(self):
        """Food codes of milk drinks that are treated as beverages (beverage mask)."""
        # Names from data/mappings/food_to_nutrient.csv ("not in table": code absent from it).
        fluid_dairy_codes = {
            "11100000",  # Milk, NFS
            "11513550",  # Chocolate milk, made from syrup with reduced fat milk
            "11436000",  # Yogurt, liquid
            "11513000",  # Chocolate milk, made from dry mix, NS as to type of milk
            "11513100",  # Chocolate milk, made from dry mix with whole milk
            "11513200",  # Chocolate milk, made from dry mix with low fat milk
            "11513300",  # Chocolate milk, made from dry mix with fat free milk
            "11513400",  # Chocolate milk, made from syrup, NS as to type of milk
            "11513500",  # Chocolate milk, made from syrup with whole milk
            "11513600",  # Chocolate milk, made from syrup with low fat milk
            "11513700",  # Chocolate milk, made from syrup with fat free milk
            "11513800",  # Chocolate milk, made from light syrup, NS as to type of milk
            "11513900",  # not in table
            "11514000",  # not in table
            "11514100",  # Hot chocolate / Cocoa, made with dry mix and water
            "11514200",  # not in table
            "11514300",  # not in table
            "11514400",  # not in table
            "11514500",  # not in table
            "11514600",  # not in table
            "11514700",  # not in table
        }
        return fluid_dairy_codes
    
    def _get_realistic_serving_sizes(self):
        """Generate realistic serving sizes based on food type and typical portions"""
        serving_sizes = np.ones(self.D, dtype=np.float32) * 100.0  # Default fallback
        
        # Define realistic serving sizes by food category/type
        # These are based on typical USDA serving sizes and common portions
        serving_size_map = {
            # Beverages (ml/g)
            "beverages": 240.0,  # 8 fl oz
            "alcoholic_beverages": 150.0,  # 5 fl oz wine, 12 fl oz beer
            "soups": 250.0,  # 1 cup
            
            # Dairy
            "dairy": 120.0,  # 1/2 cup milk, 1 slice cheese
            "milk": 240.0,  # 1 cup
            "yogurt": 170.0,  # 6 oz container
            "cheese": 30.0,  # 1 oz
            
            # Grains
            "grains": 80.0,  # 1/2 cup cooked
            "bread": 28.0,  # 1 slice
            "cereal": 30.0,  # 1 oz dry
            "rice": 80.0,  # 1/2 cup cooked
            "pasta": 80.0,  # 1/2 cup cooked
            
            # Proteins
            "protein": 85.0,  # 3 oz
            "meat": 85.0,  # 3 oz
            "fish": 85.0,  # 3 oz
            "eggs": 50.0,  # 1 large egg
            "nuts": 30.0,  # 1 oz
            
            # Fruits
            "fruits": 150.0,  # 1 medium fruit or 1/2 cup
            "apples": 150.0,
            "bananas": 120.0,
            "oranges": 130.0,
            
            # Vegetables
            "vegetables": 80.0,  # 1/2 cup cooked
            "leafy_greens": 30.0,  # 1 cup raw
            "potatoes": 150.0,  # 1 medium potato
            
            # Fats/Oils
            "fats_oils": 15.0,  # 1 tbsp
            "butter": 15.0,
            "oil": 15.0,
            
            # Condiments/Sauces
            "condiments_sauces": 15.0,  # 1 tbsp
            "ketchup": 15.0,
            "mustard": 5.0,
            "mayo": 15.0,
            
            # Snacks
            "snacks": 30.0,  # 1 oz
            "chips": 30.0,
            "crackers": 15.0,
            
            # Mixed dishes (larger portions)
            "mixed_dishes": 200.0,  # 1 cup
            "sandwiches": 150.0,  # 1 sandwich
            "pizza": 100.0,  # 1 slice
        }
        
        # Apply serving sizes based on food groups if available
        if hasattr(self, 'food_to_group') and self.food_to_group:
            unique_groups = len(set(self.food_to_group.values()))
            if self.debug:
                print(f"[DEBUG] Using food groups for serving sizes: {unique_groups} unique groups found")
            for i, food_code in enumerate(self.feature_names):
                group = self.food_to_group.get(str(food_code), "").lower()
                
                # Try exact match first
                if group in serving_size_map:
                    serving_sizes[i] = serving_size_map[group]
                else:
                    # Try partial matches
                    for category, size in serving_size_map.items():
                        if category in group or group in category:
                            serving_sizes[i] = size
                            break
                
                # Special cases based on food name and group
                name = self.code2name.get(str(food_code), "").lower()
                group = self.food_to_group.get(str(food_code), "").lower()
                text = f"{name} {group}"
                if "milk" in text:
                    serving_sizes[i] = 240.0
                elif "bread" in text:
                    serving_sizes[i] = 28.0
                elif "egg" in text:
                    serving_sizes[i] = 50.0
                elif "cheese" in text:
                    serving_sizes[i] = 30.0
                elif "cereal" in text:
                    serving_sizes[i] = 30.0
                elif "pizza" in text:
                    serving_sizes[i] = 100.0
                elif "sandwich" in text:
                    serving_sizes[i] = 150.0
        
        # Debug: show if we're using food groups or defaults
        if self.debug:
            if not (hasattr(self, 'food_to_group') and self.food_to_group):
                print("[DEBUG] No food groups available, using default 100g serving sizes")
            else:
                print(f"[DEBUG] Serving sizes range: {serving_sizes.min():.1f}g - {serving_sizes.max():.1f}g")
        
        return serving_sizes

    def _scale_energy_to_target(self, grams_i, idx, *, e_idx, E_target,
                                freeze_mask=None, per_item_max=None, iters=3,
                                respect_nutrient_caps=True, nutrient_caps=None):
        """
        Scale only adjustable items to hit E_target, keeping frozen items fixed.
        - grams_i: full D-length vector (will be modified in-place and returned)
        - idx: indices of present items in this meal
        - freeze_mask: boolean mask over idx (True=frozen at current grams)
        - per_item_max: optional scalar max grams for adjustable (non-beverage) items
        - respect_nutrient_caps: if True, avoid scaling items that would violate nutrient caps
        - nutrient_caps: dict of nutrient_name -> cap_value for constraint checking
        """
        import numpy as np

        if e_idx is None or self.nutr_per_g is None:
            return grams_i

        dens = self.nutr_per_g[idx, e_idx]  # kcal/g
        g    = grams_i[idx].copy()

        if freeze_mask is None:
            freeze_mask = np.zeros_like(g, dtype=bool)
        adj_mask = ~freeze_mask

        # nothing adjustable → return as is
        if not np.any(adj_mask):
            return grams_i

        # If respecting nutrient caps, identify items that can be safely scaled
        if respect_nutrient_caps and nutrient_caps and self.nutr_per_g is not None:
            safe_to_scale = np.ones_like(adj_mask, dtype=bool)
            
            for nut_name, cap_value in nutrient_caps.items():
                if cap_value <= 0:
                    continue
                    
                # Find nutrient index
                nut_idx = None
                for j, name in enumerate(self.nutrient_names):
                    if name == nut_name:
                        nut_idx = j
                        break
                if nut_idx is None:
                    continue
                
                # Check which items would violate the cap if scaled up
                nut_dens = self.nutr_per_g[idx, nut_idx]  # nutrient per gram
                current_nut = nut_dens @ g
                
                if current_nut > cap_value * 0.9:  # Already close to cap
                    # Find items with high nutrient density that shouldn't be scaled
                    subset = nut_dens[adj_mask]
                    if subset.size >= 2:
                        thr = np.percentile(subset, 75)
                        high_dens_mask = nut_dens > thr
                    else:
                        high_dens_mask = np.zeros_like(nut_dens, dtype=bool)
                    safe_to_scale &= ~high_dens_mask
            
            # Only scale items that are safe for all nutrients
            adj_mask = adj_mask & safe_to_scale
            if not np.any(adj_mask):
                # If no items are safe to scale, allow small energy slack instead
                E_now = float(dens @ g)
                if abs(E_now - E_target) / max(E_target, 1e-6) > 0.1:  # >10% off
                    # Apply small uniform scaling to all adjustable items
                    adj_mask = ~freeze_mask
                    s = min(1.1, E_target / max(E_now, 1e-6))
                    g[adj_mask] *= s
                    
                    # Still enforce per-item max for non-beverages even in fallback
                    if per_item_max is not None:
                        non_bev_local = (~self._bev_mask[idx]) if (self._bev_mask is not None) else np.ones_like(adj_mask, bool)
                        hit_cap = adj_mask & non_bev_local & (g > per_item_max)
                        if np.any(hit_cap):
                            g[hit_cap] = per_item_max
                    
                    grams_i[idx] = np.maximum(g, 0.0)
                    return grams_i
                else:
                    return grams_i

        for _ in range(int(iters)):
            E_now   = float(dens @ g)
            if E_now <= 1e-6:
                break
            # Energy from frozen part (implicit in g)
            E_adj   = float(dens[adj_mask] @ g[adj_mask])
            if E_adj <= 1e-6:
                break

            # multiplicative scaling on adjustable subset only
            s = (E_target - float(dens[freeze_mask] @ g[freeze_mask])) / max(E_adj, 1e-6)

            # If s is too extreme, gently clip to avoid oscillation
            s = float(np.clip(s, 0.5, 2.0))

            g[adj_mask] *= s

            # Optional per-item cap for solids
            if per_item_max is not None:
                # Identify solids among adjustable (non-beverages)
                if self._bev_mask is not None:
                    non_bev_local = ~self._bev_mask[idx]
                else:
                    non_bev_local = np.ones_like(adj_mask, dtype=bool)
                hit_cap = adj_mask & non_bev_local & (g > per_item_max)
                if np.any(hit_cap):
                    g[hit_cap] = per_item_max
                    # freeze those that hit the cap and re-run
                    adj_mask[hit_cap] = False
                    if not np.any(adj_mask):
                        break

        grams_i[idx] = np.maximum(g, 0.0)
        return grams_i

    def _cap_and_redistribute_beverages(self, grams_row, energy_idx=None, E_target=None, meal_idx=None):
        """
        Cap beverage grams and optionally redistribute energy to non-beverages.
        
        Args:
            grams_row: np.array (D,) of gram amounts
            energy_idx: index of energy nutrient in nutrient matrix
            E_target: target energy for the meal
            meal_idx: meal index for meal-specific limits (0=breakfast, 1=lunch, 2=dinner)
            
        Returns:
            np.array (D,) with capped beverages and optionally redistributed energy
        """
        if self._bev_mask is None:
            return grams_row
            
        # Get meal-specific limit or fallback to default
        if meal_idx is not None and meal_idx in self.max_beverage_grams_by_meal:
            max_bev_grams = self.max_beverage_grams_by_meal[meal_idx]
        else:
            max_bev_grams = self.max_beverage_grams
            
        if max_bev_grams <= 0:
            return grams_row
            
        g = grams_row.copy()
        
        # 1) Cap beverages by grams
        bev = self._bev_mask
        g[bev] = np.minimum(g[bev], max_bev_grams)
        
        # 2) Always apply energy share cap when requested
        if (self.bev_energy_frac_max is not None and 
            E_target is not None and energy_idx is not None and
            self.nutr_per_g is not None):
            E_bev = float(self.nutr_per_g[bev, energy_idx] @ g[bev])
            if E_bev > self.bev_energy_frac_max * E_target and E_bev > 0:
                scale = (self.bev_energy_frac_max * E_target) / E_bev
                g[bev] *= max(min(scale, 1.0), 0.0)
        
        # 3) Optionally keep energy by redistributing to non-beverages
        non = ~bev
        if E_target is not None and energy_idx is not None and np.any(g[non] > 0):
            # Compute current energy and scale non-bev to re-hit target
            E_now = float(self.nutr_per_g[:, energy_idx] @ g) if self.nutr_per_g is not None else None
            if (E_now is not None) and (E_now > 1e-6) and (abs(E_now - E_target)/max(E_target, 1e-6) > 0.005):
                # Calculate energy from non-beverages only
                E_non_bev = float(self.nutr_per_g[non, energy_idx] @ g[non])
                if E_non_bev > 1e-6:
                    # Scale non-beverages to make up the energy difference
                    energy_needed = E_target - E_now
                    scale_factor = (E_non_bev + energy_needed) / E_non_bev
                    g[non] = np.maximum(g[non] * scale_factor, 0.0)
                
        # 4) Safety: enforce per-item cap for solids post-redistribution
        if self._bev_mask is not None and getattr(self, "max_solid_grams_per_item", None) is not None:
            solids_mask = ~self._bev_mask
            g[solids_mask] = np.minimum(g[solids_mask], self.max_solid_grams_per_item)
        
        # 5) Apply group caps
        g = self._apply_group_caps_row(g)
                
        return g

    def _expand_allowed_by_subcat(self, allowed_mask, *, gate_mask=None):
        if not self.expand_by_subcategory or not self._subcat_of:
            return allowed_mask
        cats = {self._subcat_of[j] for j in np.where(allowed_mask)[0]}
        extra = np.zeros_like(allowed_mask)
        for c in cats:
            if c in self._subcat_index:
                extra[self._subcat_index[c]] = True
        if gate_mask is not None:
            extra &= gate_mask          # keep expansion inside gate
        return allowed_mask | extra

    @staticmethod
    def _align_nutrient_matrix_to_features(W_per100g, feature_ids, feature_names, nutrient_names):
        """
        Align nutrient matrix rows to match feature_names order.
        
        This is CRITICAL: nutrients_per_gram must be row-aligned with feature_names.
        If harmonized data has different row order, we must reorder to match.
        
        Args:
            W_per100g: Nutrient matrix from harmonized data (D_harmonized, P)
            feature_ids: Food codes from harmonized data (D_harmonized,)
            feature_names: Target feature names (D,)
            nutrient_names: Nutrient names (P,)
            
        Returns:
            nutrients_per_gram: Aligned matrix (D, P) in per-gram units
        """
        import numpy as np
        
        # Convert per-100g to per-gram
        W_per_gram = W_per100g / 100.0
        
        # Build mapping from feature_name to row index in harmonized data
        feature_to_row = {}
        for i, fid in enumerate(feature_ids):
            feature_to_row[str(fid)] = i
        
        # Features without a row in the harmonized data
        missing_features = []

        for i, feature_name in enumerate(feature_names):
            if str(feature_name) not in feature_to_row:
                missing_features.append(feature_name)
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing from harmonized data: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            print("Filling missing features with zero nutrient values (will be excluded from selection)")
        
        # Reorder matrix to match feature_names order, filling missing with zeros
        nutrients_per_gram = np.zeros((len(feature_names), len(nutrient_names)), dtype=np.float32)
        for i, feature_name in enumerate(feature_names):
            if str(feature_name) in feature_to_row:
                nutrients_per_gram[i, :] = W_per_gram[feature_to_row[str(feature_name)], :]
        
        # Validate final shape
        expected_shape = (len(feature_names), len(nutrient_names))
        if nutrients_per_gram.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch after alignment: {nutrients_per_gram.shape} != {expected_shape}. "
                f"Expected (D={len(feature_names)}, P={len(nutrient_names)})"
            )
        
        # Sanity check: energy should be reasonable after alignment
        if 'energy_kcal' in nutrient_names:
            energy_idx = nutrient_names.index('energy_kcal')
            energy_per_g = np.nanmean(nutrients_per_gram[:, energy_idx])
            if energy_per_g > 50:  # Likely still per-100g data
                raise ValueError(
                    f"Energy density after alignment is {energy_per_g:.1f} kcal/g, "
                    f"expected ~1-6 kcal/g. Check unit conversion."
                )
        
        return nutrients_per_gram

    def _get_pair_allowed_gate_or_none(self):
        """Load pair_allowed_gate if present, else None."""
        try:
            lyr = self.model.decoder.get_layer("pair_allowed_gate")
            return lyr.get_weights()[0]  # shape (K, D)
        except Exception:
            return None

    def _apply_calibration(self, logits, C_pair, M):
        """Apply training-time calibration in logit space."""
        # logits: (N, D)
        if hasattr(self.model, "_apply_all_presence_cal"):
            try:
                return self.model._apply_all_presence_cal(logits, C_pair, M)
            except Exception:
                pass
        return logits

    def _apply_calibration_and_mask(self, logits, C_pair, M):
        """Apply calibration and gating if available."""
        import numpy as np
        
        # 1) calibration in logit space
        logits = self._apply_calibration(logits, C_pair, M)

        # 2) hard mask using the gate (mask AFTER calibration so masked items stay zero)
        if self._gate is not None:
            gate_mask = self._gate[C_pair]                 # (N, D), 1=disallowed
            logits = np.where(gate_mask < 0.5, logits, -np.inf)

        # 3) back to probabilities
        probs = self._sigmoid_np(logits)
        
        
        return probs

    def _decode_presence_probs_from_z(self, z, C_pair, M):
        """
        Presence probabilities (N, D) for latent z, pair codes C_pair and meal indices M:
        model.predict_heads_from_z() output (probabilities are taken back to logits) or, if that
        is unavailable or fails, the decoder output; then _apply_calibration_and_mask().
        """
        import numpy as np
        # Safety check: ensure C_pair indices are within capacity
        try:
            K_embed = int(self.model.decoder.get_layer("cluster_embedding_dec").weights[0].shape[0])
        except Exception:
            K_embed = None
        K_gate = int(self._gate.shape[0]) if getattr(self, "_gate", None) is not None else None
        K_cap = K_gate if K_gate is not None else K_embed
        if K_cap is not None:
            bad = np.where((C_pair < 0) | (C_pair >= K_cap))[0]
            if bad.size:
                raise ValueError(f"C_pair contains out-of-range indices for capacity={K_cap}: rows {bad[:5]}")
        
        # Prefer modern API - ensure correct order [z, C_pair, M]
        if hasattr(self.model, "predict_heads_from_z"):
            try:
                out = self.model.predict_heads_from_z(z, C_pair, M, batch_size=256)
                # Process the output and return probabilities
                if isinstance(out, (list, tuple, np.ndarray)):
                    t = None
                    if isinstance(out, (list, tuple)):
                        for cand in out:
                            if isinstance(cand, np.ndarray) and cand.ndim == 2 and cand.shape[0] == z.shape[0]:
                                t = cand; break
                    else:
                        t = out

                    if isinstance(t, np.ndarray) and t.ndim == 2:
                        if np.all((t >= -1e-6) & (t <= 1.0 + 1e-6)):
                            return self._apply_calibration_and_mask(self._logit_np(t), C_pair, M)
                        else:
                            return self._apply_calibration_and_mask(t, C_pair, M)
            except Exception:
                # Fall back to calling the decoder directly
                pass

        # Fallback: direct decoder call
        try:
            logits = self.model.decoder.predict([z, C_pair, M], verbose=0)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return self._apply_calibration_and_mask(logits, C_pair, M)
        except Exception as e:
            raise RuntimeError(f"Decoder prediction failed; refusing to return random presences. Error: {e}")


    def _apply_temp_and_nucleus(self, P, temp=1.0, top_p=1.0):
        """Apply temperature and nucleus sampling to probabilities."""
        import numpy as np
        
        # temp in logit space
        logits = self._logit_np(P)
        if temp is not None and float(temp) > 0 and float(temp) != 1.0:
            logits = logits / float(temp)
        P = self._sigmoid_np(logits)

        # nucleus/top-p over multi-label probs (heuristic)
        if top_p < 1.0:
            idx = np.argsort(-P, axis=1)
            ps  = np.take_along_axis(P, idx, axis=1)
            csum = np.cumsum(ps, axis=1)
            keep = (csum <= top_p)
            keep[:, :1] = True  # always keep the argmax
            ps = np.where(keep, ps, 0.0)
            ps_sum = ps.sum(axis=1, keepdims=True)
            ps = np.divide(ps, np.maximum(ps_sum, 1e-8))
            P = np.zeros_like(P)
            np.put_along_axis(P, idx, ps, axis=1)
        return P

    def _select_items(self, P, n_vec, *, diversity_lambda=1.0, seed=7, max_beverages=1):
        """
        Boolean selection mask (N, D) from presence probabilities P (N, D).

        Per row i: foods with P >= 0.5 among the allowed ones (gate row of self._last_pairs[i],
        foods with nutrient data, optionally widened by subcategory); if none, the most likely
        allowed food, preferring a non-beverage. Solids are then added, most likely first (unseen
        subcategories first when prefer_diverse_subcats), up to
        min_solid_items_by_meal[self._last_meals[i]], and only the max_beverages most likely
        beverages are kept.

        n_vec, diversity_lambda and seed are not used. generate() still draws `seed` from
        self.rng, which advances that stream, so the draw is part of the reproducible sequence.
        """
        import numpy as np
        N, D = P.shape
        out = np.zeros((N, D), dtype=bool)
        threshold = 0.5

        for i in range(N):
            gate_mask = (self._gate[int(self._last_pairs[i])] < 0.5) if hasattr(self, "_last_pairs") else None
            allowed_mask = gate_mask
            if self._mappable_mask is not None:
                allowed_mask = allowed_mask & self._mappable_mask if allowed_mask is not None else self._mappable_mask
            allowed_mask = self._expand_allowed_by_subcat(allowed_mask, gate_mask=gate_mask)
            allowed_idx = np.arange(D) if allowed_mask is None else np.flatnonzero(allowed_mask)
            if allowed_idx.size == 0:
                continue

            # base selection by fixed threshold
            selected_mask = P[i] >= threshold
            if allowed_mask is not None:
                selected_mask &= allowed_mask

            # prefer at least one NON-beverage if nothing selected
            if not np.any(selected_mask):
                if (self._bev_mask is not None) and np.any(allowed_mask & ~self._bev_mask):
                    cand = np.flatnonzero(allowed_mask & ~self._bev_mask)
                    best = cand[np.argmax(P[i, cand])]
                else:
                    best = allowed_idx[np.argmax(P[i, allowed_idx])]
                selected_mask[best] = True

            # ---- enforce minimum SOLID (non-beverage) items per meal ----
            meal_idx = int(self._last_meals[i]) if hasattr(self, "_last_meals") else 0
            min_solid = int(self.min_solid_items_by_meal.get(meal_idx, 2))
            solids_mask_global = ~self._bev_mask if (self._bev_mask is not None) else np.ones(D, dtype=bool)

            solids_allowed = solids_mask_global if allowed_mask is None else (allowed_mask & solids_mask_global)
            solids_selected = selected_mask & solids_mask_global
            n_solids = int(solids_selected.sum())

            if n_solids < min_solid:
                need = min_solid - n_solids
                pool = np.flatnonzero(solids_allowed & ~selected_mask)
                if pool.size > 0:
                    # (optional) bias toward new subcategories for diversity
                    if self.prefer_diverse_subcats and (self._subcat_of is not None):
                        chosen_subcats = set(self._subcat_of[j] for j in np.where(solids_selected)[0])
                        # rank: unseen subcat first, then probability
                        unseen = np.array([self._subcat_of[j] not in chosen_subcats for j in pool], dtype=bool)
                        # take from unseen first
                        pool_unseen = pool[unseen]
                        pool_seen   = pool[~unseen]
                        order_unseen = pool_unseen[np.argsort(-P[i, pool_unseen])]
                        order_seen   = pool_seen[np.argsort(-P[i, pool_seen])]
                        add_seq = np.concatenate([order_unseen, order_seen])
                    else:
                        add_seq = pool[np.argsort(-P[i, pool])]

                    take = add_seq[:need]
                    selected_mask[take] = True
                    n_solids += len(take)

                # if gate leaves < min_solid solids available, select all available and move on

            # beverage COUNT cap (retain top-P beverages)
            if (max_beverages is not None) and (self._bev_mask is not None):
                bev_sel = np.flatnonzero(selected_mask & self._bev_mask)
                if bev_sel.size > int(max_beverages):
                    keep = bev_sel[np.argsort(P[i, bev_sel])[-int(max_beverages):]]
                    drop = np.setdiff1d(bev_sel, keep, assume_unique=False)
                    selected_mask[drop] = False

            out[i] = selected_mask

            if i < 3:
                # Get cluster name for debug output
                meal_idx = int(self._last_meals[i]) if hasattr(self, "_last_meals") else 0
                cluster_id = int(self.pair_table[int(self._last_pairs[i])][1]) if hasattr(self, "_last_pairs") else 0
                cluster_name = self._get_cluster_name(meal_idx, cluster_id)
                
                # Get food categories for selected items
                selected_indices = np.where(selected_mask)[0]
                food_categories = []
                for idx in selected_indices[:5]:  # Show up to 5 items
                    food_code = self.feature_names[idx]
                    category = self._get_food_category(food_code)
                    food_categories.append(f"{food_code}({category})")
                
                categories_str = ", ".join(food_categories)
                if len(selected_indices) > 5:
                    categories_str += f" +{len(selected_indices)-5} more"
                
                if self.debug:
                    print(f"[DEBUG] sample {i} [{cluster_name}]: sel={int(selected_mask.sum())} solids={int((selected_mask & solids_mask_global).sum())} "
                          f"(min_solid={min_solid})  maxP={P[i, selected_mask].max() if np.any(selected_mask) else 0:.3f}")
                    print(f"  Foods: {categories_str}")

        return out


    # ---------- portioning ----------

    def _portion_rdi_per_kcal(
        self, present_mask, *, min_grams=5, jitter_sigma=0.10,
        use_energy_targets=True,   # False: energy target = 2000 kcal x meal_energy_split[meal]
        meal_indices=None,         # meal indices for meal-specific beverage limits
        max_total_grams=900.0      # soft cap on total grams per meal
    ):
        """
        Grams (N, D) for the foods selected in present_mask (N, D).

        Row i takes its meal index from self._last_meals[i] (0 if unset) for the energy target
        E = meal_energy_targets[meal] (600 kcal if missing), and from meal_indices[i] for the
        per-meal beverage cap. Per row:
          1. start from serving sizes times N(1, jitter_sigma) noise drawn from self.rng
             (clipped to 0.1-3), at least min_grams;
          2. refine_portions_rdi_per_kcal(): signed-log2 steps toward the per-kcal RDI targets x E;
          3. cap solids at max_solid_grams_per_item and apply the food-group caps;
          4. rescale the adjustable items (not beverages, condiments or items at the floor) to E
             (if only fruits/vegetables are adjustable, E becomes at most 1.15x their current
             energy), holding back items dense in nutrients near their rdi_per_kcal caps;
          5. reduce sodium, total/added sugars and saturated fat greedily to their targets x E;
          6. cap beverages (grams and energy share) and the meal total at max_total_grams,
             rescaling to E after each cap.
        Without nutrient data only step 1 runs, followed by the food-group and beverage caps.
        """
        import numpy as np

        N, D = present_mask.shape
        grams = np.zeros((N, D), dtype=np.float32)

        if self.nutr_per_g is None or not self.nutrient_names:
            # Fallback: simple serving-size baseline with group/beverage caps
            for i in range(N):
                idx = np.where(present_mask[i])[0]
                if idx.size == 0:
                    continue
                g0 = self.serving_grams_vec[idx].astype(np.float32)
                if jitter_sigma > 0:
                    j = self.rng.normal(1.0, jitter_sigma, idx.size)
                    g0 *= np.clip(j, 0.1, 3.0)
                g0 = np.maximum(g0, min_grams)
                grams_i = np.zeros(D, dtype=np.float32)
                grams_i[idx] = g0
                grams_i = self._apply_group_caps_row(grams_i)
                if self._bev_mask is not None:
                    grams_i = self._cap_and_redistribute_beverages(grams_i, energy_idx=None, E_target=None, meal_idx=None)
                grams[i] = grams_i
            return grams

        name2idx = {n: i for i, n in enumerate(self.nutrient_names)}
        e_idx = name2idx.get(self.energy_name)
        if e_idx is None:
            # Same simple fallback when energy is unavailable
            for i in range(N):
                idx = np.where(present_mask[i])[0]
                if idx.size == 0:
                    continue
                g0 = self.serving_grams_vec[idx].astype(np.float32)
                if jitter_sigma > 0:
                    j = self.rng.normal(1.0, jitter_sigma, idx.size)
                    g0 *= np.clip(j, 0.1, 3.0)
                g0 = np.maximum(g0, min_grams)
                grams_i = np.zeros(D, dtype=np.float32)
                grams_i[idx] = g0
                grams_i = self._apply_group_caps_row(grams_i)
                if self._bev_mask is not None:
                    grams_i = self._cap_and_redistribute_beverages(grams_i, energy_idx=None, E_target=None, meal_idx=None)
                grams[i] = grams_i
            return grams

        # Use the pre-built rdi_per_kcal_vec
        rdi_per_kcal_vec = self._rdi_per_kcal_vec

        for i in range(N):
            idx = np.where(present_mask[i])[0]
            if idx.size == 0:
                continue

            # 1) Initial portions: canonical serving sizes (+ jitter) with floor
            g0 = self.serving_grams_vec[idx].astype(np.float32)
            if jitter_sigma > 0:
                j = self.rng.normal(1.0, jitter_sigma, idx.size)
                g0 *= np.clip(j, 0.1, 3.0)
            g0 = np.maximum(g0, min_grams)
            
            # Debug: Show initial grams before optimization
            if i < 3 and self.debug:  # debug info for first few meals
                print(f"[DEBUG] meal {i}: initial grams before optimization:")
                for j, food_idx in enumerate(idx):
                    food_code = self.feature_names[food_idx]
                    food_name = self.code2name.get(food_code, food_code)
                    category = self._get_food_category(food_code)
                    print(f"  {food_name} [{food_code}] ({category}): {g0[j]:.1f}g")

            # 2) Get energy target for this meal
            if use_energy_targets:
                meal_idx = int(self._last_meals[i]) if hasattr(self, "_last_meals") else 0
                E_meal = float(self.meal_energy_targets.get(meal_idx, 600.0))
            else:
                # Use daily energy split
                E_daily = 2000.0  # Default daily energy
                meal_idx = int(self._last_meals[i]) if hasattr(self, "_last_meals") else 0
                meal_frac = self.meal_energy_split.get(meal_idx, 0.35)
                E_meal = E_daily * meal_frac

            # 3) Refine portions using signed log2 optimization
            try:
                # Zero weight energy to avoid double counting since we rescale to exact target
                weights_under = (self.nutrient_weights or {}).copy()
                weights_over = (self.nutrient_weights or {}).copy()
                weights_under.setdefault('energy_kcal', 0.0)
                weights_over.setdefault('energy_kcal', 0.0)

                # Debug: which nutrients are being optimized + top deviations at baseline
                if i < 3 and self.debug:
                    print(f"[DEBUG] meal {i}: optimizing {len(self.nutrient_names)} nutrients (energy will be retargeted)")

                    # Per-meal targets from densities
                    targets = rdi_per_kcal_vec * E_meal  # (P,)
                    current = g0 @ self.nutr_per_g[idx, :]  # (P,) using baseline grams

                    # Exclude energy from "top deviations" since we hard-retarget it
                    skip = {self.energy_name}
                    valid = np.array([ (n not in skip) and (targets[j] > 0) for j, n in enumerate(self.nutrient_names) ])

                    ratios = np.zeros_like(current, dtype=np.float64)
                    ratios[valid] = current[valid] / np.maximum(targets[valid], 1e-8)

                    log2_ratio = np.zeros_like(ratios)
                    log2_ratio[valid] = np.log2(np.maximum(ratios[valid], 1e-8))
                    abs_log2 = np.abs(log2_ratio)

                    order = np.argsort(-abs_log2)[:5]
                    tops = []
                    for j in order:
                        if not valid[j]:
                            continue
                        name = self.nutrient_names[j]
                        tops.append(f"{name}: {current[j]:.1f} vs {targets[j]:.1f} ({log2_ratio[j]:+.2f} log2×)")
                    if tops:
                        print("  Top deviations (baseline): " + ", ".join(tops))

                g_refined = refine_portions_rdi_per_kcal(
                    grams_init=g0,
                    nutrients_per_gram=self.nutr_per_g[idx, :],  # (k, P) - already per-gram
                    nutrient_names=self.nutrient_names,
                    rdi_per_kcal_vec=rdi_per_kcal_vec,
                    energy_target=E_meal,
                    energy_name=self.energy_name,
                    max_iters=120,
                    lr=0.25,
                    weights_under=weights_under,
                    weights_over=weights_over
                )
            except Exception as e:
                # Fallback to initial portions if refinement fails
                print(f"Warning: RDI refinement failed for meal {i}: {e}")
                g_refined = g0

            # 3b) Enforce per-item cap on solids after refine (even if later scaling doesn't run)
            if self._bev_mask is not None:
                is_solid = ~self._bev_mask[idx]
            else:
                is_solid = np.ones_like(idx, bool)
            g_refined[is_solid] = np.minimum(g_refined[is_solid], self.max_solid_grams_per_item)
            
            # 3c) Apply group caps after refine
            g_refined = self._apply_group_caps_row(np.where(np.arange(self.D)[:,None]==np.array(idx), g_refined, 0).sum(axis=1))[idx]

            # 4) Build freeze mask: keep beverages, floor-stuck items, and condiments frozen
            floor_mask_local = (g_refined <= (min_grams + 1e-6))
            bev_local = self._bev_mask[idx] if self._bev_mask is not None else np.zeros_like(idx, dtype=bool)
            cond_local = np.array([self._is_condiment_idx(j) for j in idx], dtype=bool)
            freeze_mask = floor_mask_local | bev_local | cond_local

            grams_i = np.zeros(D, dtype=np.float32)
            grams_i[idx] = g_refined

            # 4a) Relax energy target if adjustable items are all fruits/vegetables
            adj_local = ~freeze_mask
            adj_groups = {self._group_of_idx(j) for j in np.array(idx)[adj_local]}
            if adj_groups and adj_groups.issubset({"fruits", "vegetables"}):
                # Keep within +15% of current energy instead of forcing full target
                E_now = float(self.nutr_per_g[idx, e_idx] @ grams_i[idx])
                E_ceiling = E_now * 1.15
                if E_ceiling < E_meal:
                    if self.debug:
                        print(f"[DEBUG] meal {i}: relaxing energy target from {E_meal:.1f} → {E_ceiling:.1f} (fruit/veg-only adjustable)")
                    E_meal = E_ceiling

            # 4b) Hit the energy target by scaling only adjustable, non-beverage items
            # Build nutrient caps for constraint-aware scaling
            nutrient_caps = {}
            if self.rdi_per_kcal is not None:
                for nut, dens in self.rdi_per_kcal.items():
                    if dens > 0:
                        nutrient_caps[nut] = float(dens * E_meal)
            
            grams_i = self._scale_energy_to_target(
                grams_i, idx, e_idx=e_idx, E_target=E_meal,
                freeze_mask=freeze_mask, per_item_max=self.max_solid_grams_per_item,
                respect_nutrient_caps=True, nutrient_caps=nutrient_caps
            )
            
            # Apply group caps and re-hit energy
            grams_i = self._apply_group_caps_row(grams_i)
            grams_i = self._scale_energy_to_target(
                grams_i, idx, e_idx=e_idx, E_target=E_meal,
                freeze_mask=freeze_mask, per_item_max=self.max_solid_grams_per_item,
                respect_nutrient_caps=True, nutrient_caps=nutrient_caps
            )

            # 5) Hard-cap upper-bound nutrients (greedy projection, linear-in-grams)
            g_refined = grams_i[idx].copy()  # Sync with current grams_i state
            for nut in ('sodium_mg', 'sugars_total_g', 'added_sugars_g', 'sfa_g'):
                j = name2idx.get(nut)
                if j is None: 
                    continue

                # Skip cap if no density is defined (prevents over-restriction)
                dens = rdi_per_kcal_vec[j]
                if dens <= 0:
                    continue  # No cap specified; skip this nutrient

                if nut == 'sodium_mg':
                    cap = float(dens * E_meal)                 # mg
                elif nut in ('sugars_total_g', 'added_sugars_g'):
                    cap = float(dens * E_meal)                 # grams
                elif nut == 'sfa_g':
                    cap = float(dens * E_meal)                 # grams

                for _ in range(12):
                    val = float(self.nutr_per_g[idx, j] @ g_refined)           # same units as col
                    if val <= cap * 1.00:
                        break
                    # Use contribution-aware reductions (grams * density, not just density)
                    contrib = (self.nutr_per_g[idx, j].clip(min=0)) * g_refined
                    if contrib.sum() <= 1e-8:
                        break
                    
                    # Preferentially reduce groups that drive this nutrient
                    prio = np.ones_like(contrib)
                    if nut in ("sugars_total_g", "added_sugars_g"):
                        for k, jj in enumerate(idx):
                            grp = self._group_of_idx(jj)
                            if grp in {"sugars", "snacks_sweets", "beverages"}:
                                prio[k] = 2.0
                    
                    excess = val - cap
                    step = 0.25 * excess
                    weights = (contrib * prio) / (np.sum(contrib * prio) + 1e-8)
                    g_refined = np.maximum(g_refined - step * weights, min_grams)

            grams_i[idx] = g_refined  # Write cap edits back to grams_i

            # 6) Re-hit energy target again after caps, still without touching beverages or floor-stuck items
            # Use the same nutrient caps to prevent re-blowing
            grams_i = self._scale_energy_to_target(
                grams_i, idx, e_idx=e_idx, E_target=E_meal,
                freeze_mask=freeze_mask, per_item_max=self.max_solid_grams_per_item,
                respect_nutrient_caps=True, nutrient_caps=nutrient_caps
            )
            
            # Apply group caps and re-hit energy
            grams_i = self._apply_group_caps_row(grams_i)
            grams_i = self._scale_energy_to_target(
                grams_i, idx, e_idx=e_idx, E_target=E_meal,
                freeze_mask=freeze_mask, per_item_max=self.max_solid_grams_per_item,
                respect_nutrient_caps=True, nutrient_caps=nutrient_caps
            )

            # 7) Apply beverage capping and redistribute energy
            meal_idx = meal_indices[i] if meal_indices is not None else None
            grams_i = self._cap_and_redistribute_beverages(
                grams_i, energy_idx=e_idx, E_target=E_meal, meal_idx=meal_idx
            )
            
            # 7b) Re-hit energy after beverage redistribution to respect max_solid_grams_per_item
            grams_i = self._scale_energy_to_target(
                grams_i, idx, e_idx=e_idx, E_target=E_meal,
                freeze_mask=freeze_mask, per_item_max=self.max_solid_grams_per_item,
                respect_nutrient_caps=True, nutrient_caps=nutrient_caps
            )
            
            # Apply group caps and re-hit energy
            grams_i = self._apply_group_caps_row(grams_i)
            grams_i = self._scale_energy_to_target(
                grams_i, idx, e_idx=e_idx, E_target=E_meal,
                freeze_mask=freeze_mask, per_item_max=self.max_solid_grams_per_item,
                respect_nutrient_caps=True, nutrient_caps=nutrient_caps
            )
            
            # 8) Apply soft total grams cap
            total_grams = float(np.sum(grams_i))
            if total_grams > max_total_grams:
                scale_factor = max_total_grams / total_grams
                grams_i = np.maximum(grams_i * scale_factor, 0.0)
                if i < 3 and self.debug:  # debug info for first few meals
                    print(f"[DEBUG] meal {i}: total grams capped from {total_grams:.0f}g to {np.sum(grams_i):.0f}g")
                
                # Re-hit energy target after grams cap (respecting the same freeze & caps)
                grams_i = self._scale_energy_to_target(
                    grams_i, idx, e_idx=e_idx, E_target=E_meal,
                    freeze_mask=freeze_mask, per_item_max=self.max_solid_grams_per_item,
                    respect_nutrient_caps=True, nutrient_caps=nutrient_caps
                )
            
            # Final group caps application
            grams_i = self._apply_group_caps_row(grams_i)
            
            grams[i] = grams_i
            
            # Debug: Show final grams after optimization
            if i < 3 and self.debug:  # debug info for first few meals
                print(f"[DEBUG] meal {i}: final grams after optimization:")
                for j, food_idx in enumerate(idx):
                    food_code = self.feature_names[food_idx]
                    food_name = self.code2name.get(food_code, food_code)
                    category = self._get_food_category(food_code)
                    final_grams = grams_i[food_idx]
                    initial_grams = g0[j]
                    change = final_grams - initial_grams
                    change_pct = (change / initial_grams * 100) if initial_grams > 0 else 0
                    print(f"  {food_name} [{food_code}] ({category}): {initial_grams:.1f}g → {final_grams:.1f}g ({change:+.1f}g, {change_pct:+.1f}%)")

        return grams

    # ---------- pair sampling ----------
    def _resolve_meal_to_idx(self, meal):
        """Convert meal name/idx to integer index."""
        if isinstance(meal, str):
            return self.meal_type_mapping.get(meal, 0)
        return int(meal)

    def _choose_pairs(self, n, *, meal=None, clusters=None, pair_codes=None):
        """Choose pair codes for generation."""
        import numpy as np
        
        if pair_codes:
            pool = [int(p) for p in pair_codes if int(p) in self.pair_table]
            if len(pool) >= int(n):
                # Use exact multiset if provided and sufficient
                sel = np.array(pool[:int(n)], dtype=np.int32)
                # shuffle to randomize order without changing counts
                self.rng.shuffle(sel)
            else:
                # Fall back to sampling from the pool
                sel = self.rng.choice(pool, size=int(n), replace=True)
        else:
            # Build pool based on meal/cluster constraints
            pool = []
            for pc, (mi, cid) in self.pair_table.items():
                if meal is not None:
                    target_mi = self._resolve_meal_to_idx(meal)
                    if mi != target_mi:
                        continue
                if clusters is not None:
                    if cid not in clusters:
                        continue
                pool.append(pc)
            
            if not pool:
                raise ValueError("No valid (meal,cluster) pairs match the selection.")
            
            sel = self.rng.choice(pool, size=int(n), replace=True)
        
        C = np.asarray(sel, dtype=np.int32)
        M = np.asarray([self.pair_table[int(p)][0] for p in C], dtype=np.int32)
        return C, M

    # ---------- main generation ----------
    def generate(self, n, *, meal=None, clusters=None, pair_codes=None,
                 temperature=1.0, nucleus_p=1.0, max_beverages=1, diversity_lambda=1.0,
                 portion_strategy="rdi_per_kcal", min_grams=5, jitter_sigma=0.15,
                 return_tidy=False, seed=None):
        """
        Generate n meals (steps in the module docstring).

        meal (name or index) and clusters (original cluster ids) restrict the sampled pairs;
        pair_codes, if given, are used as listed when at least n are valid and sampled from
        otherwise. seed re-seeds self.rng. portion_strategy and diversity_lambda are accepted but
        not used: portioning is always _portion_rdi_per_kcal().

        Returns:
            Wide DataFrame, one row per meal: grams per feature plus 'cluster_id', 'meal_idx'
            and 'pair_code'. With return_tidy=True, one row per selected food (meal_idx,
            cluster_id, food_code, grams, pair_code).
        """
        import numpy as np
        import pandas as pd
        
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        
        n = int(n)
        if n <= 0:
            return pd.DataFrame(columns=self.feature_names + ['cluster_id', 'meal_idx'])
        
        # pick pairs
        C_pair, M = self._choose_pairs(n, meal=meal, clusters=clusters, pair_codes=pair_codes)
        
        # store chosen pair codes so _select_items can read them
        self._last_pairs = np.asarray(C_pair, dtype=np.int32)  # for selection-time allowed-mask
        self._last_meals = np.asarray(M, dtype=np.int32)  # for energy-based portioning

        # sample z and decode probs
        z = self.rng.standard_normal((n, self.zdim)).astype(np.float32)
        P = self._decode_presence_probs_from_z(z, C_pair, M)
        # Optional: quick one-time debug for the first few samples
        if getattr(self, "_gate", None) is not None:
            for j in range(min(3, len(C_pair))):
                allowed = int(np.sum(self._gate[int(C_pair[j])] < 0.5))
                if self.debug:
                    print(f"[DEBUG] pair_code={int(C_pair[j])} allowed_items={allowed}")
        P = self._apply_temp_and_nucleus(P, temp=float(temperature), top_p=float(nucleus_p))

        # select items; the seed drawn here is unused by _select_items but advances self.rng
        present_mask = self._select_items(P, None, diversity_lambda=diversity_lambda, seed=int(self.rng.integers(0, 10_000_000)), max_beverages=max_beverages)

        # portion selected items (rdi_per_kcal only)
        grams = self._portion_rdi_per_kcal(
            present_mask, min_grams=min_grams, jitter_sigma=jitter_sigma, meal_indices=M,
            use_energy_targets=True
        )

        # build output
        if return_tidy:
            # Tidy format: one row per food item
            rows = []
            for i in range(n):
                present = np.where(present_mask[i])[0]
                for j in present:
                    rows.append({
                        'meal_idx': int(M[i]),
                        'cluster_id': int(self.pair_table[int(C_pair[i])][1]),
                        'food_code': self.feature_names[j],
                        'grams': float(grams[i, j]),
                        'pair_code': int(C_pair[i])
                    })
            return pd.DataFrame(rows)
        else:
            # Wide format: one row per meal
            df = pd.DataFrame(grams, columns=self.feature_names)
            df['cluster_id'] = [int(self.pair_table[int(pc)][1]) for pc in C_pair]
            df['meal_idx'] = M.astype(int)
            df['pair_code'] = C_pair.astype(int)
            return df

    # ---------- nutrition analysis helpers ----------
    @staticmethod
    def per_meal_rdi_from_energy(RDI_day, meal_type, nutrient_names, meal_fraction_map):
        """Compute per-meal RDI targets with energy-derived sugars if present."""
        import numpy as np
        
        # Handle case where RDI_day might be a dict or array
        if isinstance(RDI_day, dict):
            # Convert dict to array in the order of nutrient_names
            rdi = np.array([RDI_day.get(n, 0.0) for n in nutrient_names], dtype=float)
        else:
            rdi = np.array(RDI_day, float)
            
        frac = float(meal_fraction_map.get(meal_type, 0.33))
        
        # Find energy index
        try:
            jE = nutrient_names.index("energy_kcal")
            E_meal = rdi[jE] * frac
        except ValueError:
            # If no energy_kcal in nutrient_names, use default
            E_meal = 2000.0 * frac
            
        # proportional split baseline
        rdi *= frac
        
        # energy-derived targets (only for nutrients not in RDI data; sfa_g comes from the CSV)
        if "sugars_total_g" in nutrient_names:
            # sugars_total_g is not in the CSV, so use energy-derived calculation
            rdi[nutrient_names.index("sugars_total_g")] = 0.10 * E_meal / 4.0
        return rdi

    def show_generated_breakdown(self, n, meal_label, W, nutrient_names, RDI_day,
                                portion_strategy="rdi_per_kcal", max_rows=5, seed=123):
        """
        Generate n meals and tabulate nutrients against per-meal targets for the first max_rows.

        Args:
            n: number of meals to generate
            meal_label: meal type name; lower-cased before the meal_type_mapping lookup
            W: nutrition matrix (D x M) - nutrients per 100g
            nutrient_names: list of nutrient names (length M)
            RDI_day: daily RDI values (length M)
            portion_strategy: portioning strategy to use
            max_rows: max number of meals to show in breakdown
            seed: random seed
            
        Returns:
            df: generated meals DataFrame
            breakdown: nutrition breakdown table
        """
        import numpy as np
        import pandas as pd

        # Normalize meal label to handle case variations
        key = str(meal_label).strip().lower()

        df = self.generate(
            n,
            meal=key,                   # use normalized meal key here
            portion_strategy=portion_strategy,
            seed=seed,
            return_tidy=False
        )

        # pull grams matrix (columns are self.feature_names in the same order)
        X = df[self.feature_names].to_numpy(dtype=float)  # (n, D)

        # nutrients per meal (W is per 100g)
        N_meal = (X / 100.0) @ W                          # shape (n, M)
        col = {name: j for j, name in enumerate(nutrient_names)}

        # per-meal targets (total sugars energy-derived)
        meal_fraction_map = {
            "breakfast": self.meal_energy_split.get(0, 0.25),
            "lunch":     self.meal_energy_split.get(1, 0.35),
            "dinner":    self.meal_energy_split.get(2, 0.40),
        }
        targets = self.per_meal_rdi_from_energy(RDI_day, key, nutrient_names, meal_fraction_map)

        # choose a few readable nutrients to print (auto-skip if missing)
        keys = [k for k in ["energy_kcal","protein_g","fiber_g","sodium_mg","sfa_g","sugars_total_g",
                           "calcium_mg","iron_mg","vitamin_c_mg","vitamin_a_mcg","potassium_mg"] if k in col]

        rows = []
        for i in range(min(max_rows, len(df))):
            row = {
                "row": i,
                "pair_code": int(df.loc[i, "pair_code"]),
                "meal_idx": int(df.loc[i, "meal_idx"]),
                "cluster_id": int(df.loc[i, "cluster_id"]),
                "total_grams": float(np.sum(X[i]))
            }
            for k in keys:
                v = float(N_meal[i, col[k]])
                tgt = float(targets[col[k]])
                cov = v / max(tgt, 1e-8)
                diff = v - tgt
                row[f"{k}"] = v
                row[f"{k}_target"] = tgt
                row[f"{k}_diff"] = diff
                row[f"{k}_cov"] = cov
            rows.append(row)

        breakdown = pd.DataFrame(rows)
        # nice rounding
        for c in breakdown.columns:
            if c.endswith("_cov"):
                breakdown[c] = breakdown[c].round(2)
            elif c.endswith("_target") or c.endswith("_diff"):
                breakdown[c] = breakdown[c].round(1)
            elif c in keys or c == "total_grams":
                breakdown[c] = breakdown[c].round(1)
        return df, breakdown


# ---------- RDI per kcal refinement functions ----------

def refine_portions_log2_asymmetric(
    grams_init: np.ndarray,
    W_per100g: np.ndarray,
    nutrient_names: List[str],
    targets_vec: np.ndarray,
    energy_name: str = "energy_kcal",
    weights_under: Optional[Dict[str, float]] = None,
    weights_over: Optional[Dict[str, float]] = None,
    upper_bound_only: Optional[set] = None,
    max_iters: int = 120,
    lr: float = 0.25,
    min_grams: float = 5.0
) -> np.ndarray:
    """
    Refine portions to minimize signed log2 deviation with asymmetric penalties.
    
    Args:
        grams_init: Initial portion sizes (D,)
        W_per100g: Nutrient matrix per 100g (D, P)
        nutrient_names: List of nutrient names (P,)
        targets_vec: Target amounts per meal (P,)
        energy_name: not used
        weights_under: Weights for under-consumption penalties (default 1.0 per nutrient)
        weights_over: Weights for over-consumption penalties (default 1.0 per nutrient)
        upper_bound_only: Set of nutrients with upper-bound-only constraints
        max_iters: Maximum optimization iterations
        lr: Learning rate
        min_grams: Minimum portion size
        
    Returns:
        Refined portion sizes (D,)
    """
    import numpy as np
    
    grams = grams_init.copy()
    D, P = W_per100g.shape
    
    # Default weights
    weights_under = weights_under or {}
    weights_over = weights_over or {}
    upper_bound_only = upper_bound_only or set()
    
    # Convert per-100g to per-gram for optimization
    W_per_gram = W_per100g / 100.0
    
    for iteration in range(max_iters):
        # Current nutrient amounts
        current_nutrients = grams @ W_per_gram  # (P,)
        
        # Compute signed log2 deviations
        ratios = current_nutrients / np.maximum(targets_vec, 1e-8)
        log2_ratios = np.log2(np.clip(ratios, 1e-8, np.inf))
        
        # Compute gradients for each nutrient
        total_grad = np.zeros(D)
        
        for j, nutrient in enumerate(nutrient_names):
            if targets_vec[j] <= 0:
                continue
                
            ratio = ratios[j]
            log2_ratio = log2_ratios[j]
            
            # Determine penalty weights
            if nutrient in upper_bound_only:
                # Only penalize over-consumption
                if ratio > 1.0:
                    weight = weights_over.get(nutrient, 1.0)
                    penalty = weight * log2_ratio
                else:
                    penalty = 0.0
            else:
                # Asymmetric penalties
                if ratio < 1.0:
                    weight = weights_under.get(nutrient, 1.0)
                    penalty = weight * log2_ratio  # negative for under-consumption
                else:
                    weight = weights_over.get(nutrient, 1.0)
                    penalty = weight * log2_ratio  # positive for over-consumption
            
            # Gradient contribution from this nutrient
            # d/dg[log2(ratio)] = d/dg[log2((g^T W_j) / target_j)] = W_j / (g^T W_j * ln(2))
            current_amount = current_nutrients[j]
            if current_amount > 1e-8:
                grad_contrib = penalty * W_per_gram[:, j] / (current_amount * np.log(2))
                total_grad += grad_contrib
        
        # Update portions
        grams -= lr * total_grad
        
        # Enforce constraints
        grams = np.maximum(grams, min_grams)
        
        # Check convergence
        if np.linalg.norm(total_grad) < 1e-6:
            break
    
    return grams


def refine_portions_rdi_per_kcal(
    grams_init: np.ndarray,
    nutrients_per_gram: np.ndarray,
    nutrient_names: List[str],
    rdi_per_kcal_vec: np.ndarray,
    energy_target: float,
    *,
    energy_name: str = "energy_kcal",
    max_iters: int = 120,
    lr: float = 0.25,
    weights_under: Optional[Dict[str, float]] = None,
    weights_over: Optional[Dict[str, float]] = None,
    upper_bound_only: Optional[set] = None
) -> np.ndarray:
    """
    Refine portions for RDI per kcal strategy using signed log2 optimization.
    
    Args:
        grams_init: Initial portion sizes (D,)
        nutrients_per_gram: Nutrient matrix per gram (D, P) - NOTE: must be per-gram!
        nutrient_names: List of nutrient names (P,)
        rdi_per_kcal_vec: RDI per kcal targets (P,) (Generator._build_rdi_per_kcal_vec sets energy to 1)
        energy_target: Target energy for this meal
        energy_name: passed through; not used
        max_iters: Maximum optimization iterations
        lr: Learning rate
        weights_under: Weights for under-consumption penalties; the defaults below apply only
            when None or empty
        weights_over: Weights for over-consumption penalties; the defaults below apply only
            when None or empty
        upper_bound_only: Set of nutrients with upper-bound-only constraints

    Returns:
        Refined portion sizes (D,)
    """
    # Build per-meal targets from densities
    targets = rdi_per_kcal_vec * energy_target  # (P,)
    
    # Convert per-gram to per-100g for the helper function
    W_per100g = nutrients_per_gram * 100.0
    
    # Default weights
    weights_under = weights_under or {
        "protein_g": 2.0, "fiber_g": 2.0, "calcium_mg": 2.0, "magnesium_mg": 2.0,
        "potassium_mg": 2.0, "vitamin_d_mcg": 3.0, "vitamin_a_mcg": 1.5,
        "vitamin_k_mcg": 1.5, "folate_mcg": 1.5
    }
    weights_over = weights_over or {
        "sodium_mg": 4.0, "sfa_g": 4.0, "added_sugars_g": 3.0, "sugars_total_g": 3.0
    }
    upper_bound_only = upper_bound_only or {"sodium_mg", "added_sugars_g", "sugars_total_g", "sfa_g"}
    
    # Reuse the signed-log2 asymmetric optimizer
    return refine_portions_log2_asymmetric(
        grams_init=grams_init,
        W_per100g=W_per100g,
        nutrient_names=nutrient_names,
        targets_vec=targets,
        energy_name=energy_name,
        weights_under=weights_under,
        weights_over=weights_over,
        upper_bound_only=upper_bound_only,
        max_iters=max_iters,
        lr=lr
    )
