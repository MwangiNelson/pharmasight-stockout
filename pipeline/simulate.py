"""Generate a synthetic, deliberately messy pharmacy extract.

The hospital's records cannot leave the hospital, so this script writes the
tables a real extract would contain, with the recording problems such
extracts have:

- receipts recorded in packs instead of base units
- dispensing lines entered in error and later reversed
- the same line exported twice
- days when the dispensing system was down and lines were never entered
- receipts entered days after the stock arrived
- unrecorded losses, corrected only at monthly stock-takes
- items added to the formulary part-way through the period
- items with no opening balance
- censored demand: patients turned away during a stockout leave no record

Stock is replenished by a weekly order-up-to policy that relies on recorded,
censored dispensing. Supplier lead times vary, some orders are part-filled,
and national shortages delay or cancel orders for a few items at a time.

It also writes ``_truth.csv``, the true stock on hand, which only a
simulation can know. The pipeline uses it only to audit label quality, never
for features or training.

Run: python -m pipeline.simulate
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pipeline.config import END, N_ITEMS, RAW_DIR, SEED, START

PACK_SIZES = {
    "tablet": [28, 30, 100],
    "capsule": [30, 100],
    "injection": [1, 10],
    "syrup": [1],
    "sachet": [20],
}


def simulate(seed: int = SEED, n_items: int = N_ITEMS, start=START, end=END) -> dict:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end)
    n_days = len(days)

    # Store-wide recording outages: dispensing continues, lines are not entered.
    outage = np.zeros(n_days, bool)
    for s in rng.choice(np.arange(60, n_days - 10), 4, replace=False):
        outage[s : s + rng.integers(2, 6)] = True

    # National shortages hit a few items at a time.
    shortage = np.zeros((n_items, n_days), bool)
    for _ in range(8):
        s = rng.integers(0, n_days - 30)
        shortage[rng.random(n_items) < 0.12, s : s + rng.integers(20, 75)] = True

    # Some months the budget freezes and no orders go out in the last week.
    freeze = np.zeros(n_days, bool)
    for month_end in days[days.is_month_end]:
        if rng.random() < 0.3:
            freeze[(days > month_end - pd.Timedelta(days=7)) & (days <= month_end)] = True

    no_opening = set(rng.choice(n_items, 4, replace=False))
    items, opening, lines, moves, pos, truth = [], [], [], [], [], []
    line_id = po_id = 0
    doy, weekend = days.dayofyear.values, days.dayofweek.values >= 5
    hours = pd.to_timedelta(rng.uniform(7, 21, size=10**6), unit="h")  # dispensing clock times

    for i in range(n_items):
        item = f"MED-{i + 1:03d}"
        form = str(rng.choice(list(PACK_SIZES)))
        pack = int(rng.choice(PACK_SIZES[form]))
        lam = rng.lognormal(3.0, 0.9) * (0.3 if form == "injection" else 1.0)
        season = rng.uniform(0.2, 0.6) if rng.random() < 0.3 else 0.0
        mean = (
            lam
            * (1 + season * np.cos(2 * np.pi * (doy - rng.uniform(0, 365)) / 365))
            * (1 + rng.uniform(-0.25, 0.25) * np.arange(n_days) / n_days)
            * np.where(weekend, 0.55, 1.0)
        )
        demand = rng.negative_binomial(3, 3 / (3 + mean))  # overdispersed
        lead_mu, safety = rng.uniform(5, 21), rng.uniform(1.3, 2.0)
        review_dow, rx = rng.integers(0, 7), max(1.0, lam / rng.uniform(2, 8))
        first = 0 if rng.random() < 0.85 else int(rng.integers(1, 180))
        items.append(dict(item_id=item, form=form, base_unit="unit", pack_size=pack))

        on_hand = book = int(lam * rng.uniform(15, 45))
        if i not in no_opening:
            opening.append(dict(item_id=item, as_of=days[first].date(), qty=book, unit="UNIT"))
        recorded = np.zeros(n_days)
        in_stock = np.zeros(n_days, bool)  # what the system showed at close
        est = lam  # pharmacist's demand estimate
        open_orders = []  # [arrive_day, qty_received, po_index] in transit
        late_entries = []  # [entry_day, qty, po_index] arrived, not yet keyed in
        cancels = {}  # day -> po_index the supplier will never fill
        open_po = []  # po indices still open in the ordering system

        for d in range(first, n_days):
            day = days[d]
            # 1. deliveries arrive; most are keyed in the same day
            for o in [o for o in open_orders if o[0] == d]:
                open_orders.remove(o)
                on_hand += o[1]
                delay = int(rng.integers(1, 8)) if rng.random() < 0.05 else 0
                late_entries.append([d + delay, o[1], o[2]])
            for e in [e for e in late_entries if e[0] == d]:
                late_entries.remove(e)
                book += e[1]
                as_packs = pack > 1 and e[1] % pack == 0 and rng.random() < 0.2
                moves.append(dict(
                    item_id=item, moved_at=day, movement_type="RECEIPT",
                    qty=e[1] // pack if as_packs else e[1], unit="PACK" if as_packs else "UNIT",
                    po_id=pos[e[2]]["po_id"],
                ))
                pos[e[2]].update(status="RECEIVED", closed_at=day.date(), qty_received=e[1])
                open_po.remove(e[2])
            if d in cancels:
                pos[cancels[d]].update(status="CANCELLED", closed_at=day.date())
                open_po.remove(cancels.pop(d))

            # 2. dispense what is there; unmet demand leaves no trace
            disp = int(min(demand[d], on_hand))
            on_hand -= disp
            if disp:
                n = max(1, round(disp / rx))
                for q in rng.multinomial(disp, [1 / n] * n):
                    if q == 0 or (outage[d] and rng.random() > 0.05):
                        continue
                    line_id += 1
                    row = dict(line_id=line_id, item_id=item, dispensed_at=day + hours[line_id % 10**6],
                               qty=int(q), reversal_of=None)
                    lines.append(row)
                    if rng.random() < 0.003:
                        lines.append(dict(row))  # exported twice
                    book -= q
                    recorded[d] += q
            if rng.random() < 0.01:  # keyed in error, reversed a few days later
                line_id += 1
                q = max(1, int(rx))
                lines.append(dict(line_id=line_id, item_id=item, dispensed_at=day + hours[line_id % 10**6],
                                  qty=q, reversal_of=None))
                lines.append(dict(line_id=line_id + 1, item_id=item,
                                  dispensed_at=day + pd.Timedelta(days=int(rng.integers(1, 4)), hours=9),
                                  qty=-q, reversal_of=line_id))
                line_id += 1

            # 3. other recorded and unrecorded stock changes
            if rng.random() < 0.04 and on_hand > 0:
                q = int(min(on_hand, lam * rng.uniform(1, 4)))
                on_hand -= q
                book -= q
                moves.append(dict(item_id=item, moved_at=day, movement_type="TRANSFER_OUT",
                                  qty=q, unit="UNIT", po_id=None))
            if rng.random() < 0.03:
                on_hand -= int(on_hand * rng.uniform(0.01, 0.05))  # breakage, loss
            if rng.random() < 0.003 and on_hand > 0:
                q = int(on_hand * rng.uniform(0.05, 0.3))
                on_hand -= q
                book -= q
                moves.append(dict(item_id=item, moved_at=day, movement_type="EXPIRY",
                                  qty=q, unit="UNIT", po_id=None))
            if day.is_month_end and rng.random() < 0.75 and on_hand != book:
                moves.append(dict(item_id=item, moved_at=day, movement_type="ADJUSTMENT",
                                  qty=on_hand - book, unit="UNIT", po_id=None))
                book = on_hand

            # 4. weekly order-up-to review, using what the system shows
            if day.dayofweek == review_dow and not freeze[d] and rng.random() > 0.07:
                # average over days the system showed stock, so stockouts don't shrink the order
                window = slice(max(first, d - 27), d + 1)
                if in_stock[window].any():
                    est = recorded[window][in_stock[window]].mean()
                gap = est * (7 + lead_mu) * safety - book - sum(pos[j]["qty_ordered"] for j in open_po)
                if gap > 0:
                    qty = math.ceil(gap / pack) * pack
                    lead, fill = rng.gamma(4, lead_mu / 4), 1.0
                    if shortage[i, d]:
                        lead *= rng.uniform(2, 4)
                        fill = 0.0 if rng.random() < 0.4 else rng.uniform(0.2, 0.6)
                    elif rng.random() < 0.12:
                        fill = rng.uniform(0.4, 0.9)
                    po_id += 1
                    pos.append(dict(po_id=po_id, item_id=item, ordered_at=day.date(), qty_ordered=qty,
                                    status="OPEN", closed_at=None, qty_received=0))
                    open_po.append(len(pos) - 1)
                    received = int(qty * fill // pack * pack)
                    if received:
                        open_orders.append([d + max(1, round(lead)), received, len(pos) - 1])
                    else:  # supplier never delivers; the order is cancelled weeks later
                        cancels[d + int(rng.integers(30, 60))] = len(pos) - 1

            in_stock[d] = book > 0
            truth.append((item, day.date(), on_hand, outage[d]))

    return {
        "items": pd.DataFrame(items),
        "opening_balances": pd.DataFrame(opening),
        "dispensing": pd.DataFrame(lines).sample(frac=1, random_state=seed),  # export order is arbitrary
        "stock_movements": pd.DataFrame(moves),
        "purchase_orders": pd.DataFrame(pos),
        "_truth": pd.DataFrame(truth, columns=["item_id", "date", "true_stock", "outage"]),
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in simulate().items():
        df.to_csv(RAW_DIR / f"{name}.csv", index=False)
        print(f"{name:18s} {len(df):>8,d} rows")


if __name__ == "__main__":
    main()
