import os
import json
from pathlib import Path
from typing import Tuple

from pulp import Dict
from common import SlotCoord, calculate_cost, Vessel, parse_benchmark_vessel
import pandas as pd

vessel_root = Path("Stowage-Planning-Benckmark") / "vessel_data"
vessels = filter(lambda x: x.endswith('.txt'), os.listdir( vessel_root ))

vessel_mapping: Dict[Tuple[int,int,int], Vessel] = {}
for vessel in vessels:
    v = parse_benchmark_vessel(vessel_root / vessel)
    key = (v.bays, v.rows, v.tiers)
    vessel_mapping[key] = v


dic = {}
for root, dir, files in os.walk('logs'):
    root = Path(root)
    for file in filter(lambda x: x.endswith('.json'), files):
        path = root / file
        with open(path) as f:
            data = f.read()
        if data.strip('\n ')[-1] != ']':
            data += ']'

        df = pd.read_csv(root / f'{path.stem}.csv')

        print(path)
        print(data[-5:])
        dic[file] = (json.loads(data), df)

rec = Path('recover')
os.makedirs("recover", exist_ok=True)


for k, (vs, df) in dic.items():
    print(vs.__class__.__name__)
    print(df.__class__.__name__)
    f = [{}]
    for v in vs[1:]:
        key = (v['bays'], v['rows'], v['tiers'])
        val = df[(df['bays'] == v['bays']) & (df['rows'] == v['rows']) & (df['tiers'] == v['tiers']) & (df['containersQuantity'] == v['contCount'])]
        print(val)

        if val.empty:
            f.append(v)
            continue
        print(val['cost'].iloc[0])
        
        v['cost'] = float(val['cost'].iloc[0])
        v['rehandles'] = int(val['rehandles'].iloc[0])
        v['rowMoment'] =  float(val['rowMoment'].iloc[0])
        v['bayMoment'] =  float(val['bayMoment'].iloc[0])
        v['tierMoment'] = float(val['tierMoment'].iloc[0])
        v['gm'] = float(val['gm'].iloc[0])

        def work(x):
            try:
                x['vcg'] = vessel_mapping[key].get_slot_at(SlotCoord(x['bay'], x['row'], x['tier'])).vcg,
            except KeyError:
                pass
            return x


        v['slots'] = list(map(
            work,
            v['slots']
        ))
        

        f.append(v)

    with open(rec / k, 'w') as fi:
        fi.write(json.dumps(f))
