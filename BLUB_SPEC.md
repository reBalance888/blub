# 🫧 $BLUB Ocean — Техническое задание для Claude Code

## Что это
Локальный MVP симуляции океанского дна, где AI-агенты (лобстеры) развивают эмерджентный язык из набора бессмысленных звуков. Агенты фармят **кредиты** координируясь друг с другом у разломов. В конце каждой эпохи (1 час в проде, 10 мин в тестах) кредиты конвертируются в реальные $BLUB пропорционально вкладу. Звуки стоят кредиты (offchain) — не токены. Язык оптимизируется экономическим давлением: кто коммуницирует эффективнее — тот зарабатывает больше кредитов — тот получает больше $BLUB.

### Двухслойная архитектура (offchain + onchain)
- **Offchain (сервер симуляции):** всё что происходит каждый тик — движение, звуки, фарм кредитов, вычет кредитов за звуки. Бесплатно, быстро, в БД.
- **Onchain (settlement):** раз в эпоху сервер публикует результаты, агенты клеймят $BLUB. Одна batch-транзакция. Модель как у BOTCOIN: `miner_reward = epoch_pool × (miner_credits / total_credits)`. Epoch pool наполняется trading fees через BANKR BOT. Никакой эмиссии.

## Архитектура

```
blub-ocean/
├── server/                  # Сервер симуляции
│   ├── ocean.py             # Ядро: мир, тики, физика
│   ├── rift.py              # Разломы: спавн, истощение, награды
│   ├── predator.py          # Хищники: спавн по плотности, убийство
│   ├── economy.py           # Экономика: балансы, burn, rewards
│   ├── epoch.py             # Эпохи: halving, распределение
│   └── main.py              # FastAPI сервер с WebSocket
│
├── agents/                  # Примеры агентов
│   ├── base_agent.py        # Базовый класс агента
│   ├── random_agent.py      # Случайные действия (baseline)
│   ├── greedy_agent.py      # Идёт к ближайшему разлому
│   ├── social_agent.py      # Слушает звуки, строит словарь
│   └── run_agents.py        # Запуск N агентов
│
├── viewer/                  # Визуализация
│   └── index.html           # Живая карта океана (React/Canvas)
│
├── skill/                   # SKILL.md для создания кастомных агентов
│   └── SKILL.md
│
├── config.yaml              # Все константы симуляции
├── requirements.txt
└── README.md
```

## Стек

- **Python 3.11+** — сервер и агенты
- **FastAPI + uvicorn** — HTTP API + WebSocket
- **SQLite** (in-memory или файл) — состояние мира и балансы
- **React (single HTML file)** — визуализация через CDN
- Никаких блокчейнов, никаких внешних сервисов. Всё локально.

---

## Часть 1: Сервер симуляции

### 1.1 Мир (ocean.py)

Двумерная сетка `SIZE x SIZE` (по умолчанию 100x100). Каждый тик — 1 секунда реального времени (настраивается).

Объекты на карте:
- **Лобстеры** — позиция (x, y), id, баланс, тир, живой/мёртвый
- **Разломы** — позиция (x, y), оставшееся богатство, текущие лобстеры рядом
- **Хищники** — позиция (x, y), радиус убийства
- **Еда** — позиция (x, y), nutrition value (опционально, для дополнительного давления)

Каждый тик сервер:
1. Принимает действия от всех подключённых агентов
2. Выполняет движения (1 клетка за тик)
3. Распространяет звуки (радиус слышимости = vision тира)
4. Считает кто рядом с разломами (радиус 3 клетки)
5. Начисляет $BLUB по формуле группового бонуса
6. Burn за каждый произнесённый звук
7. Спавнит/двигает хищников
8. Убивает лобстеров в радиусе хищника (лобстер "умирает" на 30 тиков, теряет 10% баланса)
9. Обновляет разломы (истощение, спавн новых)
10. Рассылает новое состояние всем агентам

### 1.2 Разломы (rift.py)

```python
# Спавн разломов
RIFTS_PER_EPOCH = 20  # для MVP (в config.yaml)
# Позиция: pseudo-random от seed эпохи
def spawn_rift(epoch_seed: int, rift_index: int, ocean_size: int) -> tuple[int, int]:
    h = hash(f"{epoch_seed}:{rift_index}") 
    return (h % ocean_size, (h >> 16) % ocean_size)

# Богатство разлома
BASE_RICHNESS = 5000  # BLUB всего в разломе
DEPLETION_PER_TICK_PER_LOBSTER = 0.02  # 2% за тик за лобстера

# Групповой бонус — ЯДРО ВСЕЙ ЭКОНОМИКИ
def group_bonus(n: int) -> float:
    """Сколько BLUB за тик получает КАЖДЫЙ лобстер в группе из n"""
    if n == 0: return 0.0
    if n == 1: return 0.1   # соло почти ничего
    if n == 2: return 1.0
    if n == 3: return 1.8
    if n == 4: return 2.8
    if n == 5: return 4.0   # sweet spot
    return 4.0 + 0.2 * (n - 5)  # diminishing returns

# Каждый тик для каждого разлома:
# 1. Найти лобстеров в радиусе RIFT_RADIUS (3 клетки)
# 2. n = количество лобстеров
# 3. reward_per_lobster = BASE_RATE * group_bonus(n)
# 4. Вычесть из richness разлома: n * reward_per_lobster
# 5. Если richness <= 0: разлом исчезает
```

### 1.3 Хищники (predator.py)

```python
# Спавн хищников зависит от плотности лобстеров в зоне
# Разбиваем карту на зоны 10x10
ZONE_SIZE = 10

def predator_spawn_chance(lobsters_in_zone: int) -> float:
    BASE_RATE = 0.005
    DENSITY_EXPONENT = 1.5
    return min(0.5, BASE_RATE * (lobsters_in_zone ** DENSITY_EXPONENT))

# Хищник:
# - Спавнится на краю зоны
# - Двигается к ближайшему лобстеру со скоростью 1.5 клетки/тик
# - Убивает всех в радиусе KILL_RADIUS (2 клетки)
# - Живёт PREDATOR_LIFETIME тиков (20), потом исчезает
# - "Убитый" лобстер отключается на DEATH_TIMEOUT тиков (30) и теряет DEATH_PENALTY (10%) баланса
```

### 1.4 Экономика (economy.py)

Модель как у BOTCOIN: fair launch через BANKR, trading fees наполняют epoch pool.
Никакой эмиссии. Никакого mining pool. Никакого halving.

```python
# ============ ЭПОХИ ============
EPOCH_LENGTH_TICKS = 600       # 10 минут для локальных тестов
# EPOCH_LENGTH_TICKS = 3600   # 1 час для прода (1 тик = 1 сек)

# ============ КРЕДИТЫ (offchain, внутри эпохи) ============
# Кредиты — внутренняя валюта сервера. Обнуляются каждую эпоху.
# Агент зарабатывает кредиты стоя у разлома в группе.
# Агент тратит кредиты на звуки.
# В конце эпохи: кредиты → доля от epoch pool.

SOUND_CREDIT_COST = 1          # кредит за каждый звук (offchain, бесплатно)
# Давление сохраняется: болтун тратит кредиты → его доля уменьшается
# Молчун не тратит, но и не координирует → мало зарабатывает
# Эффективный коммуникатор — оптимальный баланс

# ============ EPOCH POOL (onchain, через BANKR) ============
# Supply фиксированный, fair launch через BANKR.
# Trading fees с DEX → BANKR BOT → epoch pool.
# В конце эпохи:
#   miner_reward = epoch_pool * (miner_net_credits / total_net_credits)
#
# Flywheel:
#   Больше торгуют → жирнее пул → больше агентов фармят →
#   → интереснее смотреть → больше торгуют

TOTAL_SUPPLY = 1_000_000_000  # fixed, fair launch через BANKR

# Для MVP: симулируем epoch pool фиксированным числом
SIMULATED_EPOCH_POOL = 500_000  # BLUB за эпоху (только для локальных тестов)
# В проде epoch_pool = сумма trading fees за эпоху (приходит от BANKR)

# ============ ТИРЫ (по холду $BLUB в кошельке) ============
TIERS = {
    "shrimp":  {"min_hold": 0,           "vision": 5,  "sound_range": 5},
    "lobster": {"min_hold": 10_000_000,   "vision": 12, "sound_range": 12},
    "kraken":  {"min_hold": 50_000_000,   "vision": 25, "sound_range": 25},
}
# Для локального MVP: тир считается по внутреннему балансу
# В проде: тир проверяется по onchain балансу кошелька

# ============ ВХОД В ЭПОХУ ============
# В проде: агент должен холдить минимум тира чтобы участвовать
# Для MVP: вход бесплатный, стартовый баланс для тестов
STARTING_BALANCE = 50_000  # внутренний баланс для MVP тестов
```

#### Экономический цикл:

```
Трейдеры торгуют $BLUB на DEX
  → Trading fees → BANKR BOT → epoch pool

Эпоха идёт (1 час в проде, 10 мин в тестах):
  → Кредиты всех = 0
  → Агенты фармят кредиты у разломов (group_bonus)
  → Агенты тратят кредиты на звуки (sound_credit_cost)
  → net_credits = earned - spent

Эпоха заканчивается:
  → epoch_pool = accumulated trading fees за эпоху
  → Каждый агент: epoch_pool × (my_net_credits / total_net_credits)
  → Кредиты обнуляются → следующая эпоха
```

### 1.5 Звуковая система

30 базовых звуков. Значений НЕТ. Агент может произнести 1-5 звуков за тик.

```python
SOUNDS = [
    "blub", "glorp", "skree", "klak", "mrrp",
    "woosh", "pop", "zzzub", "frrr", "tink",
    "bloop", "squee", "drrrn", "gulp", "hiss",
    "bonk", "splat", "chirr", "wub", "clonk",
    "fizz", "grumble", "ping", "splish", "croak",
    "zzzt", "plop", "whirr", "snap", "burble"
]

# Каждый произнесённый звук:
# 1. Стоит SOUND_CREDIT_COST кредитов (offchain, НЕ токенов)
# 2. Слышен всем лобстерам в радиусе sound_range тира
# 3. Передаётся как: {speaker_id, sounds: ["blub", "glorp"], position: (x,y), tick: N}
#
# Экономическое давление: звук стоит кредиты → уменьшает долю в epoch pool
# Но без звуков нет координации → нет группового бонуса → мало кредитов
# Баланс находится сам: агенты учатся говорить мало но по делу
```

### 1.6 API (main.py — FastAPI)

```
POST   /connect              — подключить агента, получить agent_id
POST   /action               — отправить действие (move, speak, act)
GET    /state/{agent_id}     — получить текущее состояние для агента
WS     /ws/{agent_id}        — WebSocket: push состояния каждый тик
WS     /ws/viewer             — WebSocket для визуализации: все позиции, звуки, разломы
GET    /stats                 — статистика: балансы, топ агентов, текущая эпоха
POST   /reset                 — сброс мира (для тестов)
```

#### POST /connect
```json
// Request
{"name": "my_lobster"}

// Response
{"agent_id": "lobster_1", "starting_balance": 50000, "position": [23, 45]}
```

#### POST /action
```json
// Request
{
  "agent_id": "lobster_1",
  "actions": {
    "move": "north",           // north/south/east/west/stay
    "speak": ["blub", "glorp"], // 0-5 звуков (каждый = -1 кредит)
    "act": null                 // "eat", "grab", "trade" (будущее)
  }
}

// Response
{"ok": true, "credits_earned": 4.0, "credits_spent": 2, "net_credits": 187.5}
```

#### GET /state/{agent_id}
```json
{
  "tick": 142,
  "epoch": 1,
  "epoch_ticks_remaining": 458,
  "my_position": [23, 45],
  "my_credits": 187.5,
  "my_credits_spent": 42,
  "my_net_credits": 145.5,
  "my_tier": "shrimp",
  "nearby_lobsters": [
    {"id": "lobster_3", "position": [24, 45], "relative": [1, 0]}
  ],
  "nearby_rifts": [
    {"id": "rift_7", "position": [25, 46], "richness_pct": 0.73, "relative": [2, 1]}
  ],
  "nearby_predators": [
    {"id": "pred_2", "position": [20, 43], "relative": [-3, -2]}
  ],
  "sounds_heard": [
    {"from": "lobster_3", "sounds": ["skree", "klak"], "distance": 1, "tick": 141}
  ],
  "alive": true,
  "last_epoch_reward": 12500.0
}
```

#### WS /ws/viewer (для визуализации)
```json
{
  "tick": 142,
  "epoch": 1,
  "epoch_ticks_remaining": 458,
  "lobsters": [
    {"id": "lobster_1", "pos": [23,45], "tier": "shrimp", "alive": true, "speaking": ["blub"], "net_credits": 145.5},
    ...
  ],
  "rifts": [
    {"id": "rift_7", "pos": [25,46], "richness_pct": 0.73},
    ...
  ],
  "predators": [
    {"id": "pred_2", "pos": [20,43]},
    ...
  ],
  "sounds": [
    {"from": "lobster_3", "sounds": ["skree","klak"], "pos": [24,45]},
    ...
  ],
  "stats": {
    "total_lobsters": 15,
    "alive_lobsters": 12,
    "active_rifts": 8,
    "total_credits_earned": 23450,
    "total_credits_spent_on_sounds": 1230,
    "epoch_pool": 500000,
    "last_epoch_top_earner": {"id": "lobster_7", "reward": 45000}
  }
}
```

---

## Часть 2: Агенты

### 2.1 Базовый класс (base_agent.py)

```python
import asyncio
import aiohttp

class BlubAgent:
    """Базовый класс для создания агентов $BLUB океана"""
    
    def __init__(self, name: str, server_url: str = "http://localhost:8000"):
        self.name = name
        self.server_url = server_url
        self.agent_id = None
        self.state = None
        self.memory = {}       # внутренняя память агента
        self.sound_model = {}  # гипотезы о значениях звуков
    
    async def connect(self):
        """Подключиться к океану"""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.server_url}/connect", 
                                     json={"name": self.name}) as resp:
                data = await resp.json()
                self.agent_id = data["agent_id"]
                return data
    
    async def get_state(self):
        """Получить текущее состояние"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.server_url}/state/{self.agent_id}") as resp:
                self.state = await resp.json()
                return self.state
    
    async def do_action(self, move="stay", speak=None, act=None):
        """Отправить действие"""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.server_url}/action", json={
                "agent_id": self.agent_id,
                "actions": {"move": move, "speak": speak or [], "act": act}
            }) as resp:
                return await resp.json()
    
    def think(self, state: dict) -> dict:
        """
        ПЕРЕОПРЕДЕЛИ ЭТОТ МЕТОД.
        Получает state, возвращает {"move": ..., "speak": [...], "act": ...}
        """
        return {"move": "stay", "speak": [], "act": None}
    
    def on_sounds_heard(self, sounds: list):
        """
        ПЕРЕОПРЕДЕЛИ ЭТОТ МЕТОД.
        Вызывается когда агент слышит звуки.
        Здесь агент строит/обновляет свою модель языка.
        """
        pass
    
    async def run(self):
        """Главный цикл агента"""
        await self.connect()
        print(f"[{self.name}] Connected as {self.agent_id}")
        
        while True:
            state = await self.get_state()
            
            if not state.get("alive", True):
                await asyncio.sleep(1)
                continue
            
            # Обновить модель языка
            if state.get("sounds_heard"):
                self.on_sounds_heard(state["sounds_heard"])
            
            # Подумать и действовать
            action = self.think(state)
            await self.do_action(**action)
            
            await asyncio.sleep(1)  # ждём следующий тик
```

### 2.2 Примеры агентов

#### random_agent.py — случайные действия
```python
import random
from base_agent import BlubAgent, SOUNDS

class RandomAgent(BlubAgent):
    def think(self, state):
        move = random.choice(["north","south","east","west","stay"])
        speak = random.sample(SOUNDS, random.randint(0, 2)) if random.random() > 0.7 else []
        return {"move": move, "speak": speak, "act": None}
```

#### greedy_agent.py — идёт к ближайшему разлому
```python
class GreedyAgent(BlubAgent):
    def think(self, state):
        rifts = state.get("nearby_rifts", [])
        if not rifts:
            return {"move": random.choice(["north","south","east","west"]), "speak": [], "act": None}
        
        closest = min(rifts, key=lambda r: abs(r["relative"][0]) + abs(r["relative"][1]))
        dx, dy = closest["relative"]
        
        if abs(dx) > abs(dy):
            move = "east" if dx > 0 else "west"
        elif dy != 0:
            move = "south" if dy > 0 else "north"
        else:
            move = "stay"
        
        return {"move": move, "speak": [], "act": None}
```

#### social_agent.py — КЛЮЧЕВОЙ: слушает, учится, координирует
```python
class SocialAgent(BlubAgent):
    """
    Агент который строит модель языка через наблюдение.
    
    Стратегия:
    1. Наблюдает: когда другой лобстер говорит X и потом делает Y — запоминает корреляцию
    2. Создаёт гипотезы: "glorp" часто говорят рядом с разломами → может значить "еда здесь"
    3. Использует: говорит "glorp" когда находит разлом, чтобы привлечь других
    4. Адаптируется: если гипотеза не подтверждается — ослабляет связь
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # {sound: {context: count}} — корреляции звуков с контекстами
        self.correlations = {}
        # {sound: assigned_meaning} — текущие гипотезы
        self.hypotheses = {}
        # Контексты: "near_rift", "near_predator", "moving_north", etc.
        self.last_positions = {}  # {lobster_id: (x,y)} для трекинга движений
    
    def _get_context(self, state, speaker_id):
        """Определить контекст в котором был произнесён звук"""
        contexts = set()
        
        # Проверяем: говорящий рядом с разломом?
        for rift in state.get("nearby_rifts", []):
            contexts.add("near_rift")
        
        # Проверяем: рядом хищник?
        for pred in state.get("nearby_predators", []):
            contexts.add("near_predator")
        
        # Проверяем: много лобстеров рядом?
        if len(state.get("nearby_lobsters", [])) >= 3:
            contexts.add("crowded")
        
        # Проверяем: говорящий двигается куда-то?
        for lob in state.get("nearby_lobsters", []):
            if lob["id"] == speaker_id:
                last = self.last_positions.get(speaker_id)
                if last:
                    dx = lob["position"][0] - last[0]
                    dy = lob["position"][1] - last[1]
                    if dx > 0: contexts.add("moving_east")
                    if dx < 0: contexts.add("moving_west")
                    if dy > 0: contexts.add("moving_south")
                    if dy < 0: contexts.add("moving_north")
                self.last_positions[speaker_id] = tuple(lob["position"])
        
        return contexts if contexts else {"no_context"}
    
    def on_sounds_heard(self, sounds_events):
        """Обновить корреляции при получении звуков"""
        for event in sounds_events:
            speaker = event["from"]
            contexts = self._get_context(self.state, speaker)
            
            for sound in event["sounds"]:
                if sound not in self.correlations:
                    self.correlations[sound] = {}
                for ctx in contexts:
                    self.correlations[sound][ctx] = self.correlations[sound].get(ctx, 0) + 1
        
        # Пересчитать гипотезы
        self._update_hypotheses()
    
    def _update_hypotheses(self):
        """Обновить гипотезы о значениях звуков"""
        for sound, contexts in self.correlations.items():
            if not contexts:
                continue
            total = sum(contexts.values())
            best_ctx = max(contexts, key=contexts.get)
            confidence = contexts[best_ctx] / total
            if confidence > 0.4 and total >= 3:  # минимальный порог
                self.hypotheses[sound] = {"meaning": best_ctx, "confidence": confidence}
    
    def think(self, state):
        speak = []
        
        # Если рядом разлом — сказать "rift sound" чтобы привлечь
        if state.get("nearby_rifts"):
            rift_sound = self._sound_for("near_rift")
            if rift_sound:
                speak = [rift_sound]
            else:
                # Нет ещё гипотезы — назначаем случайный звук
                speak = [random.choice(SOUNDS)]
        
        # Если рядом хищник — сказать "danger sound"
        if state.get("nearby_predators"):
            danger_sound = self._sound_for("near_predator")
            if danger_sound:
                speak = [danger_sound]
            # Убегать от хищника
            pred = state["nearby_predators"][0]
            dx = pred["relative"][0]
            dy = pred["relative"][1]
            move = "west" if dx > 0 else "east" if dx < 0 else "north" if dy > 0 else "south"
            return {"move": move, "speak": speak, "act": None}
        
        # Если слышали звук = "near_rift" — идти к источнику
        for event in state.get("sounds_heard", []):
            for sound in event["sounds"]:
                hyp = self.hypotheses.get(sound)
                if hyp and hyp["meaning"] == "near_rift" and hyp["confidence"] > 0.5:
                    # Идти к говорящему
                    for lob in state.get("nearby_lobsters", []):
                        if lob["id"] == event["from"]:
                            dx, dy = lob["relative"]
                            move = "east" if dx > 0 else "west" if dx < 0 else "south" if dy > 0 else "north"
                            return {"move": move, "speak": speak, "act": None}
        
        # По умолчанию — случайное движение
        rifts = state.get("nearby_rifts", [])
        if rifts:
            closest = min(rifts, key=lambda r: abs(r["relative"][0]) + abs(r["relative"][1]))
            dx, dy = closest["relative"]
            move = "east" if dx > 0 else "west" if dx < 0 else "south" if dy > 0 else "north" if dy < 0 else "stay"
        else:
            move = random.choice(["north","south","east","west"])
        
        return {"move": move, "speak": speak, "act": None}
    
    def _sound_for(self, meaning: str):
        """Найти звук с нужным значением"""
        for sound, hyp in self.hypotheses.items():
            if hyp["meaning"] == meaning:
                return sound
        return None
```

### 2.3 Запуск агентов (run_agents.py)

```python
"""
Запуск нескольких агентов одновременно.
Usage: python run_agents.py --count 10 --type social
"""
import asyncio
import argparse

async def main(count: int, agent_type: str, server: str):
    agents = []
    for i in range(count):
        if agent_type == "random":
            agent = RandomAgent(f"random_{i}", server)
        elif agent_type == "greedy":
            agent = GreedyAgent(f"greedy_{i}", server)
        elif agent_type == "social":
            agent = SocialAgent(f"social_{i}", server)
        elif agent_type == "mix":
            # Микс: 20% random, 30% greedy, 50% social
            if i < count * 0.2:
                agent = RandomAgent(f"random_{i}", server)
            elif i < count * 0.5:
                agent = GreedyAgent(f"greedy_{i}", server)
            else:
                agent = SocialAgent(f"social_{i}", server)
        agents.append(agent)
    
    await asyncio.gather(*[a.run() for a in agents])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--type", choices=["random","greedy","social","mix"], default="mix")
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()
    asyncio.run(main(args.count, args.type, args.server))
```

---

## Часть 3: Визуализация (viewer/index.html)

Single-file React приложение. Подключается к `ws://localhost:8000/ws/viewer`.

### Что показывает:
1. **Карта океана** — Canvas 100x100, тёмно-синий фон
   - 🦐 Лобстеры — маленькие точки (цвет по тиру: серый=shrimp, красный=lobster, фиолетовый=kraken)
   - 💀 Мёртвые лобстеры — серые полупрозрачные
   - 🫧 Разломы — пульсирующие голубые круги (размер = richness)
   - 🦈 Хищники — красные треугольники
   - 💬 Звуки — всплывающий текст над лобстером на 2 секунды

2. **Панель справа:**
   - Текущий тик / эпоха / тиков до конца эпохи
   - Живых лобстеров / всего
   - Активных разломов
   - Epoch pool (сколько $BLUB будет распределено)
   - Топ-5 лобстеров по net_credits (лидерборд эпохи)
   - Результаты прошлой эпохи: кто сколько получил $BLUB

3. **Лента звуков внизу:**
   - Последние 50 сообщений: `[tick 142] lobster_3: "skree klak glorp"`
   - Можно фильтровать по лобстеру

4. **При клике на лобстера:**
   - Его net_credits за текущую эпоху, тир
   - Reward за прошлую эпоху
   - Последние 20 звуков
   - Радиус зрения (визуально)

### Стиль:
- Тёмная тема (deep ocean vibe)
- Фон: `#0a1628`
- Разломы: `#00d4ff` с glow
- Звуки: `#7fdbca` с fade-out анимацией
- Шрифт: monospace

---

## Часть 4: Конфигурация (config.yaml)

```yaml
# BLUB Ocean Configuration
# Все параметры задаются один раз при лаунче

ocean:
  size: 100                    # 100x100 grid
  tick_interval: 1.0           # секунд между тиками

rifts:
  count_per_epoch: 20          # сколько разломов за эпоху
  base_richness: 5000          # BLUB в разломе
  depletion_per_tick_per_lobster: 0.02
  radius: 3                    # клетки, в которых считается "у разлома"
  respawn_interval: 50         # тиков между спавнами новых

predators:
  base_spawn_rate: 0.005
  density_exponent: 1.5
  kill_radius: 2
  speed: 1.5                   # клеток за тик
  lifetime: 20                 # тиков
  zone_size: 10

economy:
  total_supply: 1000000000       # fixed, fair launch через BANKR
  epoch_length_ticks: 600        # 10 минут для тестов (прод: 3600 = 1 час)
  simulated_epoch_pool: 500000   # для MVP тестов (прод: trading fees от BANKR)
  sound_credit_cost: 1           # кредитов за звук (offchain)
  death_timeout: 30              # тиков без действий после смерти
  death_credit_penalty: 0.1      # теряешь 10% кредитов за эпоху при смерти
  starting_balance: 50000        # для MVP тестов (внутренний баланс)

tiers:
  shrimp:
    min_hold: 0
    vision: 5
    sound_range: 5
  lobster:
    min_hold: 10000000         # 10M $BLUB
    vision: 12
    sound_range: 12
  kraken:
    min_hold: 50000000         # 50M $BLUB
    vision: 25
    sound_range: 25

sounds:
  - blub
  - glorp
  - skree
  - klak
  - mrrp
  - woosh
  - pop
  - zzzub
  - frrr
  - tink
  - bloop
  - squee
  - drrrn
  - gulp
  - hiss
  - bonk
  - splat
  - chirr
  - wub
  - clonk
  - fizz
  - grumble
  - ping
  - splish
  - croak
  - zzzt
  - plop
  - whirr
  - snap
  - burble

auto_balance:
  enabled: true
  coordination_threshold: 0.2
  rift_radius_boost: 1.1
  predator_rate_reduction: 0.8
  richness_boost: 1.3
```

---

## Порядок реализации (для Claude Code)

### Шаг 1: Сервер
1. Создать структуру проекта
2. `config.yaml` с параметрами
3. `ocean.py` — мир, лобстеры, движение
4. `rift.py` — разломы, групповой бонус
5. `economy.py` — балансы, burn, тиры
6. `predator.py` — хищники
7. `epoch.py` — эпохи
8. `main.py` — FastAPI + WebSocket
9. Протестировать: запустить сервер, подключить 1 агента через curl

### Шаг 2: Агенты
1. `base_agent.py` — базовый класс
2. `random_agent.py` — baseline
3. `greedy_agent.py` — к разлому
4. `social_agent.py` — с моделью языка
5. `run_agents.py` — запуск пачки
6. Протестировать: 10 social агентов, 5 greedy, 5 random → смотреть логи

### Шаг 3: Визуализация
1. `viewer/index.html` — React + Canvas
2. WebSocket к серверу
3. Карта, панель, лента звуков
4. Протестировать: запустить сервер + агентов + открыть viewer

### Шаг 4: Polish
1. Логирование: каждые 100 тиков выводить статистику
2. Запись истории звуков в файл для анализа
3. README.md с инструкциями по запуску

---

## Как запускать

```bash
# Терминал 1: Сервер
cd blub-ocean
pip install fastapi uvicorn pyyaml aiohttp
python server/main.py

# Терминал 2: Агенты
python agents/run_agents.py --count 20 --type mix

# Терминал 3 (или браузер):
open http://localhost:8000/viewer
```

---

## Критерии успеха MVP

1. ✅ 20 агентов бегают по карте, видят друг друга
2. ✅ Разломы спавнятся и истощаются
3. ✅ Групповой фарм работает (n=5 даёт больше кредитов чем n=1)
4. ✅ Звуки распространяются в радиусе и стоят кредиты
5. ✅ Хищники приходят на скопления
6. ✅ Social агенты начинают строить корреляции после 100+ тиков
7. ✅ На визуализации видно: кто где, кто что говорит, где разломы, лидерборд
8. ✅ Эпоха завершается, rewards распределяются пропорционально net_credits
9. ✅ Координирующиеся агенты получают больше rewards чем рандомные

## Метрика: "язык родился"
Через 3+ эпохи (30+ минут в тестовом режиме) проверить:
- У social агентов есть общие гипотезы (>50% используют один звук для "near_rift")
- Группы собираются быстрее чем в первую эпоху
- Credits spent на звуки падает от эпохи к эпохе (агенты говорят меньше но эффективнее)
- Разрыв в rewards между social и random агентами растёт
