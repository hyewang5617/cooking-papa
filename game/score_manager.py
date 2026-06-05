import json
import os
from datetime import datetime

STEAK_RANKING_FILE   = 'rankings_steak.json'
PANCAKE_RANKING_FILE = 'rankings_pancake.json'
_LEGACY_FILE         = 'rankings.json'


class ScoreManager:
    def __init__(self):
        self.total_score = 0
        self.combo       = 0
        self.max_combo   = 0
        self.game_scores = []
        self._migrate_legacy()
        self.steak_rankings   = self._load(STEAK_RANKING_FILE)
        self.pancake_rankings = self._load(PANCAKE_RANKING_FILE)

    # ── public ────────────────────────────────────────────────────────────────

    def add_score(self, base_points):
        self.combo += 1
        self.max_combo  = max(self.max_combo, self.combo)
        multiplier      = min(1.0 + (self.combo // 4) * 0.25, 2.5)
        earned          = int(base_points * multiplier)
        self.total_score += earned
        return earned

    def add_bonus(self, points):
        self.total_score += points

    def record_game(self, name, score):
        self.game_scores.append({'game': name, 'score': score})

    def save(self, player_name, category='steak'):
        entry = {
            'name':  player_name,
            'score': self.total_score,
            'date':  datetime.now().strftime('%Y-%m-%d %H:%M'),
        }
        filename = STEAK_RANKING_FILE if category == 'steak' else PANCAKE_RANKING_FILE
        lst = self.steak_rankings if category == 'steak' else self.pancake_rankings

        lst.append(entry)
        lst.sort(key=lambda e: e['score'], reverse=True)

        seen, deduped = set(), []
        for e in lst:
            if e['name'] not in seen:
                seen.add(e['name'])
                deduped.append(e)
        deduped = deduped[:10]

        if category == 'steak':
            self.steak_rankings = deduped
        else:
            self.pancake_rankings = deduped

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(deduped, f, ensure_ascii=False, indent=2)

    def get_rankings(self, category='steak'):
        return self.steak_rankings if category == 'steak' else self.pancake_rankings

    @property
    def rankings(self):
        return self.steak_rankings

    def reset(self):
        self.total_score  = 0
        self.combo        = 0
        self.max_combo    = 0
        self.game_scores  = []
        self.steak_rankings   = self._load(STEAK_RANKING_FILE)
        self.pancake_rankings = self._load(PANCAKE_RANKING_FILE)

    # ── private ───────────────────────────────────────────────────────────────

    def _load(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _migrate_legacy(self):
        """Move old rankings.json → rankings_steak.json if not yet migrated."""
        if os.path.exists(_LEGACY_FILE) and not os.path.exists(STEAK_RANKING_FILE):
            try:
                with open(_LEGACY_FILE, encoding='utf-8') as f:
                    data = json.load(f)
                with open(STEAK_RANKING_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, IOError):
                pass
